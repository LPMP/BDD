#include "bdd_solver.h"
#include "ILP_parser.h"
#include "bdd_mma.h"
#include "decomposition_bdd_mma.h"
#include <variant>
#include <iomanip>
#include <CLI/CLI.hpp>

namespace LPMP {

    bdd_solver::bdd_solver(int argc, char** argv)
    {
        // setup command line arguemnts
        CLI::App app("LPMP BDD solver");
        app.allow_extras();
 
        std::string input_file;
        app.add_option("-i, --input_file", input_file, "ILP input file name")
            ->required()
            ->check(CLI::ExistingPath);

        std::unordered_map<std::string, ILP_input::variable_order> variable_order_map{
            {"input", ILP_input::variable_order::input},
            {"bfs", ILP_input::variable_order::bfs},
            {"cuthill", ILP_input::variable_order::cuthill},
            {"mindegree", ILP_input::variable_order::mindegree}
        };

        ILP_input::variable_order variable_order_ = ILP_input::variable_order::input;
        app.add_option("-o, --order", variable_order_, "variable order")
            ->transform(CLI::CheckedTransformer(variable_order_map, CLI::ignore_case));


        size_t max_iter = 1000;
        app.add_option("--max_iter", max_iter, "maximal number of iterations, default value = 1000")
            ->check(CLI::PositiveNumber);

        enum class bdd_solver_impl { mma, decomposition_mma } bdd_solver_impl_;
        std::unordered_map<std::string, bdd_solver_impl> bdd_solver_impl_map{
            {"mma",bdd_solver_impl::mma},
            {"decomposition_mma",bdd_solver_impl::decomposition_mma}
        };

        app.add_option("-s, --solver", bdd_solver_impl_, "the name of solver for the relaxation")
            ->required()
            ->transform(CLI::CheckedTransformer(bdd_solver_impl_map, CLI::ignore_case));

        std::variant<bdd_mma::averaging_type, decomposition_bdd_mma::options> solver_options_;

        app.callback([&app, &bdd_solver_impl_, &solver_options_]() {
                CLI::App solver_app;

                if(bdd_solver_impl_ == bdd_solver_impl::mma || bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
                {
                bdd_mma::averaging_type avg_type = bdd_mma::averaging_type::classic;
                std::unordered_map<std::string, bdd_mma::averaging_type> averaging_type_map{
                {"classic", bdd_mma::averaging_type::classic},
                {"srmp", bdd_mma::averaging_type::srmp}
                };

                solver_app.add_option("--averaging", avg_type, "min marginal averaging type")
                ->transform(CLI::CheckedTransformer(averaging_type_map, CLI::ignore_case));

                if(bdd_solver_impl_ == bdd_solver_impl::mma)
                {
                solver_app.parse(app.remaining_for_passthrough()); 
                std::cout << "chose mma solver\n";
                solver_options_ = avg_type;
                }
                else if(bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
                {
                    std::cout << "chose decomposition mma solver\n";
                    size_t nr_threads = 0;
                    solver_app.add_option("--nr_threads", nr_threads, "number of threads for simultaneous optimization of the Lagrange decomposition")
                        ->required()
                        ->check(CLI::PositiveNumber);

                    double mp_weight = 1.0;;
                    solver_app.add_option("--parallel_message_passing_weight", mp_weight , "weight for passing messages between threads")
                        ->check(CLI::Range(0.0,1.0));

                    solver_app.parse(app.remaining_for_passthrough());
                    solver_options_ = decomposition_bdd_mma::options{avg_type, nr_threads, mp_weight};
                } 
                }
                else
                    throw std::runtime_error("solver type not supported\n");
        });

        //CLI11_PARSE(app, argc, argv);

        app.parse(argc, argv);

        ILP_input ilp = get_ILP(input_file, variable_order_);
        bdd_preprocessor bdd_pre(ilp);
        bdd_pre.construct_bdd_collection();
        bdd_storage stor(bdd_pre);

        std::cout << std::setprecision(10);
        if(bdd_solver_impl_ == bdd_solver_impl::mma)
        {
            bdd_mma solver(stor, ilp.objective().begin(), ilp.objective().end());
            std::cout << "constructed mma solver\n";
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                solver.iteration();
                std::cout << "iteration " << iter << ", lower bound = " << solver.lower_bound() << "\n";
            }
        }
        else if(bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
        {
            decomposition_bdd_mma solver(stor, ilp.objective().begin(), ilp.objective().end(), std::get<decomposition_bdd_mma::options>(solver_options_)); 
            std::cout << "constructed decomposition mma solver\n";
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                solver.iteration();
                std::cout << "iteration " << iter << ", lower bound = " << solver.lower_bound() << "\n";
            }
            solver.backward_run(); // To flush out the Lagrange multiplier queues and recompute final lower bound
            std::cout << "final lower bound = " << solver.lower_bound() << "\n";
        }
        else
        {
            assert(false);
        }
    }

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

}

int main(int argc, char** argv)
{
    LPMP::bdd_solver(argc, argv);
}
