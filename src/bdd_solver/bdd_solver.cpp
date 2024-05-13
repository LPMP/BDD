#include "bdd_solver/bdd_solver.h"
#include "ILP/ILP_parser.h"
#include "ILP/OPB_parser.h"
//#include "input_parser.h"
#include "bdd_conversion/bdd_preprocessor.h"
#include "min_marginal_utils.h"
#include "bdd_solver/incremental_mm_agreement_rounding_cuda.h"
#include "bdd_solver/incremental_mm_agreement_rounding.hxx"
#include "wedelin_primal_heuristic.hxx"
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <memory>
#include <unordered_set>
#include <stdlib.h>
#include <stdexcept>
#include <filesystem>
//#include <CLI/CLI.hpp>
#include "bdd_logging.h"
#include "time_measure_util.h"
#include "run_solver_util.h"
#include "mm_primal_decoder.h"
#include <string>

using json = nlohmann::json;

namespace LPMP {

    bdd_solver::bdd_solver(const std::string &config)
    {
        config_ = read_config(config);
    }

    bdd_solver::bdd_solver(const char* config)
    {
        config_ = read_config(config);
    }

    // json config for ILP parsing:
    // "input" : filename.txt or ILP string,
    // the input field is required
    ILP_input bdd_solver::read_ILP(const nlohmann::json &config)
    {
        if (!config.contains("input"))
            throw std::runtime_error("no input specified");

        // determine whether input file or input string is specified
        const std::string input = config["input"].template get<std::string>();
        std::ifstream f(input);
        if (f)
        {
            bdd_log << "[bdd_solver] Read input file " << config["input"] << "\n";
            return ILP_parser::parse_file(input);
        }
        else // input might be ILP in string
        {
            bdd_log << "[bdd_solver] Read input string " << input << "\n";
            try {
                return ILP_parser::parse_string(input);
            } catch(...) {
                return OPB_parser::parse_string(input);
            }
        }
    }

    // json config for processing the ILP:
    // "var_order" : input|bfs|cuthill|mindegree,
    // "normalize_constraints" : true|false,
    void bdd_solver::process_ILP(const nlohmann::json& config, ILP_input &ilp)
    {
        if (config.contains("variable order"))
        {
            const std::string var_order = config["variable order"].template get<std::string>();
            if (var_order == "bfs")
            {
                bdd_log << "[bdd_solver] Reorder ilp variables with BFS ordering\n";
                ilp.reorder_bfs();
            }
            else if (var_order == "cuthill")
            {
                bdd_log << "[bdd_solver] Reorder ilp variables with Cuthill ordering\n";
                ilp.reorder_Cuthill_McKee();
            }
            else if (var_order == "minimum degree")
            {
                bdd_log << "[bdd_solver] Reorder ilp variables with minimum degree ordering\n";
                ilp.reorder_minimum_degree_ordering();
            }
            else if (var_order == "input")
            {
            }
            else
                throw std::runtime_error("Variable order " + var_order + " unknown");
        }

        if (config.contains("normalize constraints") && config["normalize constraints"].template get<bool>() == true || (config.contains("relaxation solver") && config["relaxation solver"].template get<std::string>() == "sequential mma"))
        {
            bdd_log << "[bdd_solver] Normalize constraints\n";
            ilp.normalize();
        }
    }

    // json config for transformation into BDDs:
    // "split_long_bdds": true|false,
    // "split_bdds_length": ${number},
    // "split_bdds_implication": true|false
    // if split_long_bdds is true but no maximum bdd length is given, a value is computed that will result in good GPU utilization.
    // if split_bdds_implication is true, then an additional BDD will be added that potentially propagates information between different split bdds faster.
    // 1 input: ILP_input; 1 output: bdd_collection
    BDD::bdd_collection bdd_solver::transform_to_BDDs(const nlohmann::json& config, const ILP_input& ilp)
    {
        bdd_log << "[bdd solver] Compute BDDs\n";
        const bool normalize_constraints = config.contains("normalize constraints") ? config["normalize constraints"].template get<bool>() : false;
        const bool split_bdds = config.contains("split bdds");
        // if no split length is given, a value is computed that will result in good GPU utilization
        const size_t split_length = (split_bdds && config["split bdds"].contains("split length")) ? config["split bdds"]["split length"].template get<size_t>() : std::numeric_limits<size_t>::max();
        const bool split_bdds_implication = (split_bdds && config["split bdds"].contains("implication bdd")) ? config["split bdds"]["implication"].template get<bool>() : false;
        bdd_preprocessor bdd_pre(ilp, normalize_constraints, split_bdds, split_bdds_implication, split_length);

        return bdd_pre.get_bdd_collection();
    }

    // json config for dual Lagrange relaxation solver:
    // "solver": sequential mma|parallel mma|cuda mma|cuda lbfgs parallel mma|lbfgs parallel mma|subgradient
    // "smoothing": true|false // not currently implemented
    // "precision": float|double
    // 2 input: bdd_collection, ILP_input; 1 output: solver_type
    typename bdd_solver::solver_type bdd_solver::construct_solver(const nlohmann::json&config, const BDD::bdd_collection& bdd_col, const std::vector<double>& costs)
    {
        const std::string precision = config.contains("precision") ? config["precision"].template get<std::string>() : "double";
        if (!(precision == "double" || precision == "single" || precision == "float"))
            throw std::runtime_error("precision must be double|single|float");

        const std::string solver_type = config.contains("relaxation solver") ? config["relaxation solver"].template get<std::string>() : "sequential mma";

        const bool smoothing = config.contains("smoothing") ? config["smoothing"].template get<bool>() : false;

        bdd_log << "[bdd solver] Allocate relaxation solver " << solver_type << " with precision " << precision << " and " << (smoothing ? "" : "no") << " smoothing\n";

        if (solver_type == "sequential mma")
        {
            if (precision == "double")
            {
                return bdd_mma_base<bdd_branch_instruction_bdd_index<double, uint32_t>>(bdd_col, costs);
            }
            else
            {
                return bdd_mma_base<bdd_branch_instruction_bdd_index<float, uint32_t>>(bdd_col, costs);
            }
        }
        else if (solver_type == "parallel mma")
        {
            if (precision == "double")
            {
                return bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>(bdd_col, costs);
            }
            else
            {
                return bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>(bdd_col, costs);
            }
        }
#ifdef WITH_CUDA
            else if(solver_type == "cuda parallel mma")
            {
                if (precision == "double")
                {
                    return bdd_cuda_parallel_mma<float>(bdd_col, costs);
                }
                else
                {
                    return bdd_cuda_parallel_mma<double>(bdd_col, costs);
                }
            }
#endif
            else if(solver_type.find("lbfgs") == 0)
            {
                int lbfgs_history_size = lbfgs_default_history_size;
                double lbfgs_init_step_size = lbfgs_default_init_step_size;
                double lbfgs_req_rel_lb_increase = lbfgs_default_req_rel_lb_increase;
                double lbfgs_step_size_decrease_factor = lbfgs_default_step_size_decrease_factor;
                double lbfgs_step_size_increase_factor = lbfgs_default_step_size_increase_factor;

                if(config.contains("lbfgs"))
                {
                    const auto l = config["lbfgs"];
                    if(l.contains("history size"))
                        lbfgs_history_size = l["history size"].template get<size_t>();
                    if(l.contains("initial step size"))
                        lbfgs_init_step_size = l["initial step size"].template get<double>();
                    if(l.contains("required relative lb increase"))
                        lbfgs_req_rel_lb_increase = l["required relative lb increase"].template get<double>();
                    if(l.contains("step size decrease factor"))
                        lbfgs_step_size_decrease_factor = l["step size decrease factor"].template get<double>();
                    if(l.contains("step size increase factor"))
                        lbfgs_step_size_increase_factor = l["step size increase factor"].template get<double>();
                    
                }

                bdd_log << "[bdd solver] lbfgs parameters:\n";
                bdd_log << "\t\thistory size " << lbfgs_history_size << "\n";
                bdd_log << "\t\tinitial step size " << lbfgs_init_step_size << "\n";
                bdd_log << "\t\trequired relative lb increase " << lbfgs_req_rel_lb_increase << "\n";
                bdd_log << "\t\tstep size decrease factor " << lbfgs_step_size_decrease_factor << "\n";
                bdd_log << "\t\tstep size increase factor " << lbfgs_step_size_increase_factor << "\n";

                if(solver_type == "lbfgs parallel mma")
                {
                if (precision == "double")
                {
                    return lbfgs_parallel_mma_double_type(bdd_col, costs,
                    lbfgs_history_size, lbfgs_init_step_size, lbfgs_req_rel_lb_increase, lbfgs_step_size_decrease_factor, lbfgs_step_size_increase_factor);
                }
                else
                {
                    return lbfgs_parallel_mma_float_type(bdd_col, costs,
                    lbfgs_history_size, lbfgs_init_step_size, lbfgs_req_rel_lb_increase, lbfgs_step_size_decrease_factor, lbfgs_step_size_increase_factor);
                }
                }
#ifdef WITH_CUDA
                else if(solver_type == "lbfgs cuda mma")
                {
                    if (precision == "double")
                    {
                        return cuda_lbfgs_parallel_mma_double_type(bdd_col, costs,
                        lbfgs_history_size, lbfgs_init_step_size, lbfgs_req_rel_lb_increase, lbfgs_step_size_decrease_factor, lbfgs_step_size_increase_factor);
                    }
                    else
                    {
                        return cuda_lbfgs_parallel_mma_float_type(bdd_col, costs,
                        lbfgs_history_size, lbfgs_init_step_size, lbfgs_req_rel_lb_increase, lbfgs_step_size_decrease_factor, lbfgs_step_size_increase_factor);
                    }
                }
#endif
                else
                    throw std::runtime_error("lbfgs solver " + solver_type + " unknown");
            }
            else if(solver_type == "subgradient")
            {
                if (precision == "double")
                {
                    return subgradient<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>, float>(bdd_col, costs);
                }
                else
                {
                    return subgradient<bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>, double>(bdd_col, costs);
                }
            }
#ifdef WITH_CUDA
            else if(solver_type == "cuda lbfgs parallel mma")
            {
                if (precision == "double")
                {
                    return cuda_lbfgs_parallel_mma_double_type(bdd_col, costs);
                }
                else
                {
                    return cuda_lbfgs_parallel_mma_float_type(bdd_col, costs);
                }
            }
#endif
            else
            {
                throw std::runtime_error("relaxation solver " + solver_type + " unknown");
            }
    }

    // json config for running the solver:
    // "maximum iter": ${number},
    // "time limit": ${number},
    // "improvement slope": ${number},
    // "minimum improvement": ${number}
    // time limit is in seconds
    // improvement slope is a fraction stating how much improvement per iteration is minimally needed to continue optimization
    // minimum improvement is the minimum increase in lower bound to continue optimization
    void bdd_solver::solve_dual(const nlohmann::json &config, typename bdd_solver::solver_type &solver)
    {
        size_t max_iter = 1000;
        double min_improvement = 1e-6;
        double improvement_slope = 1e-9;
        double time_limit = 3600;

        if (config.contains("termination criteria"))
        {
            const auto tc = config["termination criteria"];
            if (tc.contains("maximum iterations"))
                max_iter = tc["maximum iterations"].template get<size_t>();
            if (tc.contains("minimum improvement"))
                min_improvement = tc["minimum improvement"].template get<double>();
            if (tc.contains("improvement slope"))
                improvement_slope = tc["improvement slope"].template get<double>();
            if (tc.contains("time limit"))
                time_limit = tc["time limit"].template get<double>();
        }

        bdd_log << "[bdd solver] termination criteria:\n";
        bdd_log << "\t\tmaximum iterations: " << max_iter << "\n";
        bdd_log << "\t\tminimum improvement: " << min_improvement << "\n";
        bdd_log << "\t\timprovement slope: " << improvement_slope << "\n";
        bdd_log << "\t\ttime limit: " << time_limit << "s\n";

        std::visit([&](auto &&s)
                   { run_solver(s, max_iter, min_improvement, improvement_slope, time_limit); },
                   solver);

        bdd_log << "[bdd solver] Terminated dual optimization\n";
        bdd_log << "[bdd solver] SetValue for solver\n";
    }

    // json config for rounding a primal solution with perturbation
    // "initial perturbation": ${number}
    // "perturbation growth rate": ${number}
    // "inner iterations": ${integer}
    // "outer iterations": ${integer}
    // wiring: 
    // 2 input: solver_type, ILP_input, 1 output: variable assignment vector
    std::vector<char> bdd_solver::perturbation_rounding(const nlohmann::json& config, typename bdd_solver::solver_type& solver, const ILP_input& ilp)
    {
        double init_perturbation = 0.1;
        double perturbation_growth_rate = 1.1;
        size_t nr_inner_iterations = 100;
        size_t nr_outer_iterations = 100;

        if (config.contains("perturbation rounding"))
        {
            const auto pr = config["perturbation rounding"];
            if (pr.contains("initial perturbation"))
                init_perturbation = pr["initial perturbation"];
            if (pr.contains("perturbation growth rate"))
                perturbation_growth_rate = pr["perturbation growth rate"];
            if (pr.contains("inner iterations"))
                nr_inner_iterations = pr["inner iterations"];
            if (pr.contains("outer iterations"))
                nr_outer_iterations = pr["outer iterations"];

            const auto sol = std::visit([&](auto &&s) {
                if constexpr( // CPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::sequential_mma_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::sequential_mma_double_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::parallel_mma_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::parallel_mma_double_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::lbfgs_parallel_mma_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::lbfgs_parallel_mma_double_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::subgradient_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::subgradient_double_type>
                            )
                            {
                    return incremental_mm_agreement_rounding_iter(s, init_perturbation, perturbation_growth_rate, nr_inner_iterations, nr_outer_iterations);
                            }
#ifdef WITH_CUDA
                    else if constexpr( // GPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::cuda_parallel_mma_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::cuda_lbfgs_parallel_mma_double_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::cuda_lbfgs_parallel_mma_float_type>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, typename bdd_solver::cuda_lbfgs_parallel_mma_double_type>
                            )
                    {
                        return incremental_mm_agreement_rounding_cuda(s, init_perturbation, perturbation_growth_rate, nr_inner_iterations, true, nr_outer_iterations);
                    }
#endif
                    else
                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    } },
                                        solver);

            // TODO: reset solver state to before rounding
            double obj = std::numeric_limits<double>::infinity();
            if (sol.size() >= ilp.nr_variables())
                obj = ilp.evaluate(sol.begin(), sol.begin() + ilp.nr_variables());
            bdd_log << "[incremental primal rounding] solution objective = " << obj << "\n";
            return sol;
        }
        else
        {
            return std::vector<char>{}; // set empty output if no solution was rounded
        }
    }

    void bdd_solver::print_statistics(const nlohmann::json& config, const ILP_input& ilp, const BDD::bdd_collection& bdd_col)
    {
        if (config.contains("print statistics"))
        {
            bdd_log << "[print_statistics] #variables = " << ilp.nr_variables() << "\n";
            bdd_log << "[print_statistics] #constraints = " << ilp.nr_constraints() << "\n";
            bdd_log << "[print_statistics] #BDDs = " << bdd_col.nr_bdds() << "\n";
            const auto num_constraints = ilp.nr_constraints();
            std::vector<size_t> num_constraints_per_var(ilp.nr_variables(), 0);
            for (size_t c = 0; c < ilp.nr_constraints(); c++)
                for (const auto v : ilp.variables(c))
                    num_constraints_per_var[v]++;
            bdd_log << "[print_statistics] minimum num. constraints per var = " << *std::min_element(num_constraints_per_var.begin(), num_constraints_per_var.end()) << "\n";
            bdd_log << "[print_statistics] maximum num. constraints per var = " << *std::max_element(num_constraints_per_var.begin(), num_constraints_per_var.end()) << "\n";
            bdd_log << "[print_statistics] mean num. constraints per var = " << std::accumulate(num_constraints_per_var.begin(), num_constraints_per_var.end(), 0.0) / num_constraints_per_var.size() << "\n";
        }
    }

    void bdd_solver::export_bdd_lp(const nlohmann::json& config, const BDD::bdd_collection& bdd_col, const std::vector<double>& costs)
    {
        if (config.contains("export bdd lp"))
        {
            const std::string file = config["export bdd lp"].template get<std::string>();
            std::ofstream f;
            f.open(file);
            bdd_col.write_bdd_lp(f, costs.begin(), costs.end());
            f.close();
        }
    }

    void bdd_solver::export_lp(const nlohmann::json &config, const ILP_input &ilp)
    {
        if (config.contains("export lp"))
        {
            std::ofstream f;
            const std::string file = config["export lp"].template get<std::string>();
            f.open(file);
            const std::string extension = std::filesystem::path(file).extension();
            if (extension == ".lp")
                ilp.write_lp(f);
            else if (extension == ".opb")
                ilp.write_opb(f);
            else if (extension == ".mps")
                ilp.write_mps(f);
            else
                throw std::runtime_error("Cannot recognize file extension " + extension + " for exporting problem file");
            f.close();
        }
    }

    void bdd_solver::export_bdd_graph(const nlohmann::json& config, const BDD::bdd_collection& bdd_col)
    {
        if (config.contains("export bdd graph"))
        {
            const std::string file = config["export bdd graph"];
            const auto [filename, extension] = [&]() -> std::tuple<std::string, std::string>
            {
                const auto idx = file.rfind('.');
                if (idx != std::string::npos)
                {
                    const std::string filename = file.substr(0, idx);
                    const std::string extension = file.substr(idx + 1);
                    assert(extension == "dot");
                    return {filename, extension};
                }
                else
                {
                    // no extension found
                    return {file, "dot"};
                }
            }();

            for (size_t bdd_nr = 0; bdd_nr < bdd_col.nr_bdds(); ++bdd_nr)
            {
                std::ofstream f;
                const std::string dot_file = filename + "_" + std::to_string(bdd_nr) + ".dot";
                const std::string png_file = filename + "_" + std::to_string(bdd_nr) + ".png";
                f.open(dot_file);
                bdd_col.export_graphviz(bdd_nr, f);
                f.close();
                const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
                system(convert_command.c_str());
            }
        }
    }

    nlohmann::json bdd_solver::read_config(const std::string &c)
    {
        std::ifstream f(c);
        if (!f) // input argument is not a file, try to parse it as a json string
            return json::parse(c);
        else
            return json::parse(f);
    }

    void bdd_solver::solve(nlohmann::json &config)
    {
        if (ilp_.nr_variables() == 0)
        {
            ilp_ = read_ILP(config);
            process_ILP(config, ilp_);
            export_lp(config, ilp_);

            bdd_col_ = transform_to_BDDs(config, ilp_);
            print_statistics(config, ilp_, bdd_col_);
            export_bdd_graph(config, bdd_col_);
            export_bdd_lp(config, bdd_col_, ilp_.objective());

            solver_ = construct_solver(config, bdd_col_, ilp_.objective());
        }

        solve_dual(config, solver_);
        perturbation_rounding(config, solver_, ilp_);
    }

    two_dim_variable_array<std::array<double, 2>> bdd_solver::min_marginals()
    {
        const auto mms = std::visit([&](auto &&s) -> two_dim_variable_array<std::array<double, 2>>
                                    { 
                                        if constexpr(
                    std::is_same_v<std::remove_reference_t<decltype(s)>, sequential_mma_float_type> || std::is_same_v<std::remove_reference_t<decltype(s)>, sequential_mma_double_type>
                    ||
                    std::is_same_v<std::remove_reference_t<decltype(s)>, parallel_mma_float_type> || std::is_same_v<std::remove_reference_t<decltype(s)>, parallel_mma_double_type>
                    )
                                        {
                                         return s.min_marginals();
                                        }
                                        else
                                        throw std::runtime_error("Min marginals not implemented for the given solver"); 
                                        },
                                    solver_);
        return permute_min_marginals(mms, ilp_.get_variable_permutation());
    }

    //std::tuple<std::vector<std::string>, std::vector<double>, std::vector<double>> bdd_solver::min_marginals_with_variable_names()
    //{
    //    return export_min_marginals_with_names(min_marginals(), options.ilp.var_index_to_name());
    //}

    void bdd_solver::fix_variable(const size_t var, const bool value)
    {
        std::visit([var, value](auto &&s)
                   {
            if constexpr(
                    std::is_same_v<std::remove_reference_t<decltype(s)>, sequential_mma_float_type> || std::is_same_v<std::remove_reference_t<decltype(s)>, sequential_mma_double_type>
                    ||
                    std::is_same_v<std::remove_reference_t<decltype(s)>, parallel_mma_float_type> || std::is_same_v<std::remove_reference_t<decltype(s)>, parallel_mma_double_type>
                    )
            s.fix_variable(var, value);
            else
            throw std::runtime_error("Fix variable not implemented for the given solver");
                   },
                   solver_);
    }

    void bdd_solver::fix_variable(const std::string &var, const bool value)
    {
        const size_t var_index = ilp_.get_var_index(var);
        fix_variable(var_index, value);
    }

    double bdd_solver::lower_bound()
    {
        return std::visit([](auto &&s)
                          { return s.lower_bound(); },
                          solver_);
    }

    //void bdd_solver::export_difficult_core()
    //{
    //    mm_primal_decoder mms(min_marginals());
    //    std::unordered_set<size_t> one_fixations, zero_fixations;
    //    for (size_t i = 0; i < mms.size(); ++i)
    //    {
    //        const auto mmt = mms.compute_mm_type(i);
    //        const auto mm_sum = mms.mm_sum(i);
    //        if (mmt == mm_type::one && mm_sum[1] + options.export_difficult_core_th <= mm_sum[0])
    //            one_fixations.insert(i);
    //        else if (mmt == mm_type::zero && mm_sum[0] + options.export_difficult_core_th <= mm_sum[1])
    //            zero_fixations.insert(i);
    //    }
    //    ILP_input reduced_ilp = options.ilp.reduce(zero_fixations, one_fixations);
    //    bdd_log << "[bdd solver] Difficult core has " << reduced_ilp.nr_variables() << " variables and " << reduced_ilp.constraints().size() << " constraints left\n";
//
//        std::ofstream f;
//        f.open(options.export_difficult_core);
//        const std::string extension = std::filesystem::path(options.export_difficult_core).extension();
//        if (extension == ".lp")
//            reduced_ilp.write_lp(f);
//        else if (extension == ".opb")
//            reduced_ilp.write_opb(f);
//        else if (extension == ".mps")
//            reduced_ilp.write_mps(f);
//        else
//            throw std::runtime_error("Cannot recognize file extension " + extension + " for difficult core export file");
//        f.close();
//    }

}