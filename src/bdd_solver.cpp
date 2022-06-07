#include "bdd_solver.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include "min_marginal_utils.h"
#include "incremental_mm_agreement_rounding_cuda.h"
#include "incremental_mm_agreement_rounding.hxx"
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
#include <CLI/CLI.hpp>
#include "time_measure_util.h"
#include "run_solver_util.h"
#include "mm_primal_decoder.h"
#include <string>
#include <regex>

namespace LPMP {

    std::unique_ptr<char*[]> convert_string_to_argv(const std::vector<std::string>& args)
    {
        std::unique_ptr<char*[]> argv = std::make_unique<char*[]>(args.size()+1);

        static constexpr const char* prog_name = "bdd_solver";
        argv[0] = const_cast<char*>(prog_name);
        for(size_t i=0; i<args.size(); ++i)
            argv[i+1] = const_cast<char*>(args[i].c_str());
        return argv; 
    }

    void print_statistics(ILP_input& ilp, bdd_preprocessor& bdd_pre)
    {
        std::cout << "[print_statistics] #variables = " << ilp.nr_variables() << "\n";
        std::cout << "[print_statistics] #constraints = " << ilp.nr_constraints() << "\n";
        std::cout << "[print_statistics] #BDDs = " << bdd_pre.nr_bdds() << "\n";
    }

    ILP_input parse_ilp_file(const std::string& filename)
    {
        // determine whether file is in LP format or in opb one.
        if(filename.substr(filename.find_last_of(".") + 1) == "opb")
        {
            std::cout << "[bdd solver] Parse opb file\n";
            return OPB_parser::parse_file(filename);
        }
        else if(filename.substr(filename.find_last_of(".") + 1) == "lp")
        {
            std::cout << "[bdd solver] Parse lp file\n";
            return ILP_parser::parse_file(filename);
        }
        else // peek into file
        {
            throw std::runtime_error("peeking into files not implemented yet");
        }
    }

    ILP_input parse_ilp_string(const std::string& input)
    {
        // if file begins with * (i.e. opb comment) or with min: then it is an opb file
        std::regex comment_regex("^\\w*\\*");
        std::regex opb_min_regex("^\\w*min:");
        if(std::regex_search(input, comment_regex) || std::regex_search(input, opb_min_regex)) 
        {
            std::cout << "[bdd solver] Parse opb string\n";
            return OPB_parser::parse_string(input); 
        }
        else
        {
            std::cout << "[bdd solver] Parse lp string\n";
            return ILP_parser::parse_string(input); 
        }
    }

    bdd_solver_options::bdd_solver_options(int argc, char** argv)
        : bdd_solver_options(argc, argv, parse_ilp_file, parse_ilp_string)
    {}

    bdd_solver_options::bdd_solver_options(const std::vector<std::string>& args)
        : bdd_solver_options(args.size()+1, convert_string_to_argv(args).get())
    {}

    /*
    bdd_solver::bdd_solver(int argc, char** argv)
        : bdd_solver(bdd_solver_options(argc, argv))
    {}

    bdd_solver::bdd_solver(const std::vector<std::string>& args)
        : bdd_solver(bdd_solver_options(args))
    {}
    */

    //bdd_solver::bdd_solver(bdd_solver_options&& opt)
    //    : options(opt)
    //{
    //    throw std::runtime_error("kwas");
    //}

    bdd_solver::bdd_solver(bdd_solver_options opt)
        : options(opt)
    {
        options.ilp.reorder(options.var_order);
        options.ilp.normalize();

        std::cout << "[bdd solver] ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints\n";
        if(options.ilp.preprocess())
            std::cout << "[bdd solver] ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints after preprocessing\n";
        else
        {
            std::cout << "[bdd solver] The problem appears to be infeasible." << std::endl;
            return;
        }

        const auto start_time = std::chrono::steady_clock::now();

        costs = options.ilp.objective();

        if(options.take_cost_logarithms)
        {
            std::cout << "[bdd solver] Take logarithms of costs\n";
            for(size_t i=0; i<costs.size(); ++i)
            {
                assert(costs[i] > 0);
                costs[i] = std::log(costs[i]);
            }
        }

        if(options.optimization == bdd_solver_options::optimization_type::maximization)
        {
            std::cout << "[bdd solver] Use negative costs due to maximization\n";
            for(size_t i=0; i<costs.size(); ++i)
                costs[i] = -costs[i];
        }

        const bool normalize_constraints = [&]() {
            if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::sequential_mma)
                return true;
            return false;
        }();

        bdd_preprocessor bdd_pre(options.ilp, options.constraint_groups, normalize_constraints);

        std::cout << std::setprecision(10);

        if(options.statistics)
        {
            print_statistics(options.ilp, bdd_pre);
            exit(0);
        }
        else if(!options.export_bdd_lp_file.empty())
        {
            std::ofstream f;
            f.open(options.export_bdd_lp_file);
            bdd_pre.get_bdd_collection().write_bdd_lp(f, costs.begin(), costs.end());
            f.close(); 
            exit(0);
        }
        else if(!options.export_bdd_graph_file.empty())
        {

            // get ending and filename
            const auto [filename, extension] = [&]() -> std::tuple<std::string, std::string> {
                const auto idx = options.export_bdd_graph_file.rfind('.');
                if(idx != std::string::npos)
                {
                    const std::string filename = options.export_bdd_graph_file.substr(0, idx);
                    const std::string extension = options.export_bdd_graph_file.substr(idx+1);
                    assert(extension == "dot");
                    return {filename, extension};
                }
                else
                {
                    // no extension found
                    return {options.export_bdd_graph_file, "dot"};
                }
            }();

            for(size_t bdd_nr=0; bdd_nr<bdd_pre.get_bdd_collection().nr_bdds(); ++bdd_nr)
            {
                std::ofstream f;
                const std::string dot_file = filename + "_" + std::to_string(bdd_nr) + ".dot";
                const std::string png_file = filename + "_" + std::to_string(bdd_nr) + ".png";
                f.open(dot_file);
                bdd_pre.get_bdd_collection().export_graphviz(bdd_nr, f);
                f.close(); 
                const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
                system(convert_command.c_str());
            }
            exit(0);
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::sequential_mma)
        {
            if(options.smoothing == 0)
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                std::cout << "[bdd solver] constructed sequential mma solver\n"; 
            } else {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_mma_smooth<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_mma_smooth<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                std::cout << "[bdd solver] constructed sequential smooth mma solver\n"; 
            }
        } 
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::parallel_mma)
        {
            if(options.smoothing == 0)
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_parallel_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_parallel_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                std::cout << "[bdd solver] constructed parallel mma solver\n"; 
            }
            else
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_parallel_mma_smooth<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_parallel_mma_smooth<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                std::cout << "[bdd solver] constructed smooth parallel mma solver\n"; 
            }
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_cuda)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for cuda mma");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_cuda<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_cuda<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            std::cout << "[bdd solver] constructed CUDA based mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::hybrid_parallel_mma)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for hybrid parallel mma");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_multi_parallel_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_multi_parallel_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            std::cout << "[bdd solver] constructed CUDA based mma solver\n"; 
        }
        else
        {
            throw std::runtime_error("no solver nor output of statistics or export of lp selected");
        }

        // set smoothing
        if(options.smoothing != 0.0)
            std::visit([&](auto&& s) { 
                    if constexpr(
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<double>>
                            )
                    s.set_smoothing(options.smoothing);
                    else
                    throw std::runtime_error("smoothing not implemented for chosen solver");
                    }, *solver);

        // set constant
        if(options.ilp.constant() != 0.0)
            std::visit([&](auto&& s) { 
                    if constexpr(
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<double>>
                            )
                    s.add_to_constant(options.ilp.constant());
                    else
                    throw std::runtime_error("constants not implemented for chosen solver");
            }, *solver);

        auto setup_time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000;
        std::cout << "[bdd solver] setup time = " << setup_time << " s" << "\n";
        options.time_limit -= setup_time;
    }

    void bdd_solver::solve()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        if(options.time_limit < 0)
        {
            std::cout << "[bdd_solver] Time limit exceeded." << std::endl;
            return;
        }
        std::visit([&](auto&& s) {

                run_solver(s, options.max_iter, options.tolerance, options.improvement_slope, options.time_limit);
                }, *solver);

        // TODO: improve, do periodic tightening
        if(options.tighten)
        {
            for(size_t tighten_iter=0; tighten_iter<10; ++tighten_iter)
            {
            tighten();
            std::visit([&](auto&& s) {
                    for(size_t iter=0; iter<10; ++iter)
                        s.iteration();
                    }, *solver);
            }
        }

        if(options.solution_statistics)
        {
            std::cout << "[bdd solver] print solution statistics:\n";
            const auto var_perm = options.ilp.get_variable_permutation().inverse_permutation();
            assert(var_perm.size() == options.ilp.nr_variables());
            const auto mms = min_marginals();
            const auto mm_diffs = min_marginal_differences(mms, 0.0);
            assert(mms.size() == options.ilp.nr_variables());
            assert(mm_diffs.size() == options.ilp.nr_variables());

            for(size_t i=0; i<mm_diffs.size(); ++i)
            {
                std::cout << options.ilp.get_var_name(var_perm[i]) << ", position = " << var_perm[i] << ", c = " << options.ilp.objective(var_perm[i]) << ", min marginal = " << mm_diffs[i] << "\n";
            }

            print_mm(mms);
        }

        if(options.export_difficult_core != "")
            export_difficult_core();
    }

    two_dim_variable_array<std::array<double,2>> bdd_solver::min_marginals()
    {
        const auto mms = std::visit([&](auto&& s) { 
                return s.min_marginals();
                }, *solver); 
        return permute_min_marginals(mms, options.ilp.get_variable_permutation());
    }

    double bdd_solver::round()
    {
        if(options.incremental_primal_rounding)
        {
            std::cout << "[incremental primal rounding] start rounding\n";
            const auto sol = std::visit([&](auto&& s) {
                    if constexpr( // CPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            // TODO: remove for cuda rounding again //
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            //////////////////////////////////////////
                            )
                    return incremental_mm_agreement_rounding_iter(s, options.incremental_initial_perturbation, options.incremental_growth_rate, options.incremental_primal_num_itr_lb);
                    else if constexpr( // GPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            )
                    {
                    return s.incremental_mm_agreement_rounding(options.incremental_initial_perturbation, options.incremental_growth_rate, options.incremental_primal_num_itr_lb);
                    }

                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    }
                    }, *solver);

            const double obj = options.ilp.evaluate(sol.begin(), sol.end());
            std::cout << "[incremental primal rounding] solution objective = " << obj << "\n";
            return obj;
        }
        else if(options.wedelin_primal_rounding)
        {
            std::cout << "[Wedelin primal rounding] start rounding\n";
            const auto sol = std::visit([&](auto&& s) {
                    if constexpr( // CPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            // TODO: remove for cuda rounding again //
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            //////////////////////////////////////////
                            )
                    return wedelin_rounding(s, options.ilp,
                            options.wedelin_theta,
                            options.wedelin_delta,
                            options.wedelin_kappa_min, options.wedelin_kappa_max,
                            options.wedelin_kappa_step, options.wedelin_alpha,
                            500);
                    else if constexpr( // GPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            )
                    {
                    throw std::runtime_error("Wedelin rounding not implemented for GPU solver yet");
                    }

                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    }
                    }, *solver);

            const double obj = options.ilp.evaluate(sol.begin(), sol.end());
            std::cout << "[incremental primal rounding] solution objective = " << obj << "\n";
            return obj;

        }
        else // no rounding
        {
            return std::numeric_limits<double>::infinity();
        }
    }

    void bdd_solver::tighten()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        if(options.time_limit < 0)
        {
            std::cout << "Time limit exceeded, aborting tightening." << std::endl;
            return;
        }

        std::cout << "Tighten...\n";
        std::visit([](auto&& s) {
            if constexpr(std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>)
            s.tighten();
            else
                throw std::runtime_error("tighten not implemented");
            }, *solver);
    }

    void bdd_solver::fix_variable(const size_t var, const bool value)
    {
        std::visit([var, value](auto&& s) {
            if constexpr(
                    std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                    ||
                    std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                    )
            s.fix_variable(var, value);
            else
                throw std::runtime_error("fix variable not implemented");
            }, *solver);
    }

    void bdd_solver::fix_variable(const std::string& var, const bool value)
    {
        const size_t var_index = options.ilp.get_var_index(var);
        fix_variable(var_index, value);
    }

    double bdd_solver::lower_bound()
    {
        return std::visit([](auto&& s) {
                return s.lower_bound(); 
                }, *solver);
    }

    void bdd_solver::export_difficult_core()
    {
        mm_primal_decoder mms(min_marginals());
        std::unordered_set<size_t> one_fixations, zero_fixations;
        for(size_t i=0; i<mms.size(); ++i)
        {
            const auto mmt = mms.compute_mm_type(i);
            const auto mm_sum = mms.mm_sum(i);
            if(mmt == mm_type::one && mm_sum[1] + options.export_difficult_core_th <= mm_sum[0])
                one_fixations.insert(i);
            else if(mmt == mm_type::zero && mm_sum[0] + options.export_difficult_core_th <= mm_sum[1])
                zero_fixations.insert(i);
        }
        ILP_input reduced_ilp = options.ilp.reduce(zero_fixations, one_fixations);
        std::cout << "[bdd solver] Difficult core has " << reduced_ilp.nr_variables() << " variables and " << reduced_ilp.constraints().size() << " constraints left\n";

        std::ofstream f;
        f.open(options.export_difficult_core);
        const std::string extension = std::filesystem::path(options.export_difficult_core).extension();
        if(extension == ".lp")
            reduced_ilp.write_lp(f);
        else if(extension == ".opb")
            reduced_ilp.write_opb(f);
        else
            throw std::runtime_error("Cannot recognize file extension " + extension + " for difficult core export file");
        f.close(); 

    }

}
