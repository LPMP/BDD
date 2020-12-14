#include "bdd_solver.h"
#include "ILP_parser.h"
#include <iomanip>
#include <memory>
#include <CLI/CLI.hpp>
#include "time_measure_util.h"

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

    void print_statistics(ILP_input& ilp, bdd_storage& stor)
    {
        std::cout << "#variables = " << ilp.nr_variables() << "\n";
        std::cout << "#constraints = " << ilp.nr_constraints() << "\n";
        std::cout << "#BDDs = " << stor.nr_bdds() << "\n";
        const auto var_groups = stor.compute_variable_groups();
        std::cout << "#var groups = " << var_groups.size() << "\n";
        std::cout << "#average nr vars per group = " << stor.nr_variables() / double(var_groups.size()) << "\n";

    }

    bdd_solver::bdd_solver(int argc, char** argv)
    {
        // setup command line arguemnts
        CLI::App app("LPMP BDD solver");
        app.allow_extras();
 
        auto input_group = app.add_option_group("input", "input either from file or as string");
        std::string input_file;
        auto input_file_arg = input_group->add_option("-i, --input_file", input_file, "ILP input file name")
            ->check(CLI::ExistingPath);

        std::string input_string;
        auto input_string_arg = input_group->add_option("--input_string", input_string, "ILP input in string");

        input_group->require_option(1); // either as string or as filename

        std::unordered_map<std::string, ILP_input::variable_order> variable_order_map{
            {"input", ILP_input::variable_order::input},
            {"bfs", ILP_input::variable_order::bfs},
            {"cuthill", ILP_input::variable_order::cuthill},
            {"mindegree", ILP_input::variable_order::mindegree}
        };

        ILP_input::variable_order variable_order_ = ILP_input::variable_order::input;
        app.add_option("-o, --order", variable_order_, "variable order")
            ->transform(CLI::CheckedTransformer(variable_order_map, CLI::ignore_case));


        app.add_option("--max_iter", max_iter, "maximal number of iterations, default value = 1000")
            ->check(CLI::PositiveNumber);

        enum class bdd_solver_impl { mma, mma_srmp, mma_agg, decomposition_mma, anisotropic_mma, mma_vec } bdd_solver_impl_;
        std::unordered_map<std::string, bdd_solver_impl> bdd_solver_impl_map{
            {"mma",bdd_solver_impl::mma},
            {"decomposition_mma",bdd_solver_impl::decomposition_mma},
            {"mma_srmp",bdd_solver_impl::mma_srmp},
            {"mma_agg",bdd_solver_impl::mma_agg},
            {"anisotropic_mma",bdd_solver_impl::anisotropic_mma},
            {"mma_vec",bdd_solver_impl::mma_vec}
        };

        auto solver_group = app.add_option_group("solver", "solver either a BDD solver or output of statistics");
        solver_group->add_option("-s, --solver", bdd_solver_impl_, "the name of solver for the relaxation")
            ->transform(CLI::CheckedTransformer(bdd_solver_impl_map, CLI::ignore_case));

        bool primal_rounding = false;
        app.add_flag("-p, --primal", primal_rounding, "primal rounding flag");
        int fixing_var_order_ = 1;
        std::unordered_map<std::string, int> fixing_var_order_map{{"marg_abs",0},{"marg_up",1},{"marg_down",2},{"marg_red",3}};
        int fixing_var_value_ = 0;
        std::unordered_map<std::string, int> fixing_var_value_map{{"marg",0},{"red",1},{"one",2},{"zero",3}};
        if (primal_rounding)
        {
            // TODO fix adoption of options (default value is not changed)
            app.add_option("--fixing_order", fixing_var_order_, "variable order for primal heuristic, default value = marg_up")
                ->transform(CLI::CheckedTransformer(fixing_var_order_map, CLI::ignore_case));
            app.add_option("--fixing_value", fixing_var_value_, "preferred variable value for primal heuristic, default value = marg")
                ->transform(CLI::CheckedTransformer(fixing_var_value_map, CLI::ignore_case));
        }

        bool statistics = false;
        solver_group->add_flag("--statistics", statistics, "statistics of the problem");
        solver_group->require_option(1); // either a solver or statistics

        decomposition_bdd_mma::options decomposition_mma_options;
        app.callback([&app, &bdd_solver_impl_, &decomposition_mma_options]() {
                CLI::App solver_app;

                if(bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
                {
                    std::cout << "use decomposition mma solver\n";
                    size_t nr_threads = 0;
                    solver_app.add_option("--nr_threads", nr_threads, "number of threads for simultaneous optimization of the Lagrange decomposition")
                        ->required()
                        ->check(CLI::PositiveNumber);

                    double mp_weight = 1.0;;
                    solver_app.add_option("--parallel_message_passing_weight", mp_weight , "weight for passing messages between threads")
                        ->check(CLI::Range(0.0,1.0));

                    solver_app.parse(app.remaining_for_passthrough());
                    decomposition_mma_options = decomposition_bdd_mma::options{nr_threads, mp_weight};
                } 
        });

        app.parse(argc, argv);

        ILP_input ilp = [&]() {
            if(!input_file.empty())
                return ILP_parser::parse_file(input_file);
            else if(!input_string.empty())
                return ILP_parser::parse_string(input_string);
            else
                throw std::runtime_error("could not detect ILP input");
        }();

        std::cout << "ILP has " << ilp.nr_variables() << " variables and " << ilp.nr_constraints() << " constraints\n";

        ilp.reorder(variable_order_);
        costs = ilp.objective();

        // print variable order
        // for (size_t v = 0; v < ilp.nr_variables(); v++)
        //     std::cout << ilp.get_var_name(v) << std::endl;

        bdd_preprocessor bdd_pre(ilp);
        //bdd_pre.construct_bdd_collection(); // this is only needed if bdd collection is used in bdd preprocessor
        bdd_storage stor(bdd_pre);

        std::cout << std::setprecision(10);

        if(statistics)
        {
            print_statistics(ilp, stor);
            exit(0);
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::mma)
        {
            solver = std::move(bdd_mma(stor, ilp.objective().begin(), ilp.objective().end()));
            std::cout << "constructed mma solver\n";
        } 
        else if(bdd_solver_impl_ == bdd_solver_impl::mma_srmp)
        {
            solver = std::move(bdd_mma_srmp(stor, ilp.objective().begin(), ilp.objective().end()));
            std::cout << "constructed srmp mma solver\n";
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::mma_agg)
        {
            solver = std::move(bdd_mma_agg(stor, ilp.objective().begin(), ilp.objective().end()));
            std::cout << "constructed aggressive mma solver\n";
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
        {
            solver = std::move(decomposition_bdd_mma(stor, ilp.objective().begin(), ilp.objective().end(), decomposition_mma_options));
            std::cout << "constructed decomposition mma solver\n";
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::anisotropic_mma)
        {
            solver = std::move(bdd_mma_anisotropic(stor, ilp.objective().begin(), ilp.objective().end()));
            std::cout << "constructed anisotropic mma solver\n"; 
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::mma_vec)
        {
            solver = std::move(bdd_mma_vec(stor, ilp.objective().begin(), ilp.objective().end()));
            std::cout << "constructed vectorized mma solver\n"; 
        }
        else
        {
            assert(false);
        }

        if (primal_rounding)
        {
            std::cout << "Fixing variable order: " << fixing_var_order_ << std::endl;
            std::cout << "Fixing variable value: " << fixing_var_value_ << std::endl;
            primal_heuristic = std::move(bdd_fix(stor, fixing_var_order_, fixing_var_value_));
            std::cout << "constructed primal heuristic\n";
        }
    }

    bdd_solver::bdd_solver(const std::vector<std::string>& args)
        : bdd_solver(args.size()+1, convert_string_to_argv(args).get())
    {}

    ILP_input bdd_solver::get_ILP(const std::string& input_file, ILP_input::variable_order variable_order_)
    { 
        ILP_input ilp = ILP_parser::parse_file(input_file);
        ilp.reorder(variable_order_);
        return ilp;
    }

    bdd_storage bdd_solver::transfer_to_bdd_storage(bdd_preprocessor& bdd_pre)
    {
        bdd_storage stor(bdd_pre);
        return stor;
    }

    void initialize_solver(bdd_storage& stor)
    {

    }


    void bdd_solver::solve()
    {
        return std::visit([&](auto&& s) {
                s.solve(max_iter);
            }, *solver);
    }

    void bdd_solver::round()
    {
        if (!primal_heuristic)
            return;

        std::vector<double> total_min_marginals = std::visit([](auto&& s) { return s.total_min_marginals(); }, *solver);
        bool success = primal_heuristic->round(total_min_marginals);

        if (!success)
            return;

        std::vector<char> primal_solution = primal_heuristic->primal_solution();
        double upper_bound = std::inner_product(primal_solution.begin(), primal_solution.end(), costs.begin(), 0.0);
        std::cout << "Upper bound = " << upper_bound << std::endl;
    } 

    double bdd_solver::lower_bound()
    {
        return std::visit([](auto&& s) {
                return s.lower_bound(); 
                }, *solver);

    }

}

