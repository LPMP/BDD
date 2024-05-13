#pragma once

#include "ILP/ILP_input.h"
// bdd mma
#include "bdd_mma_base.h"
#include "bdd_branch_instruction.h"
// bdd parallel mma
#include "bdd_parallel_mma_base.h"
// lbfgs
#include "lbfgs.h"
// subgradient
//#include "bdd_subgradient.h"
#include "subgradient.h"
#ifdef WITH_CUDA
#include "bdd_cuda_parallel_mma.h"
#endif

// TODO: remove
//#include "bdd_mma.h"
//#include "bdd_lbfgs_parallel_mma.h"
//#include "bdd_mma_smooth.h"
//#include "bdd_cuda.h"
//#include "bdd_parallel_mma.h"
//#include "bdd_parallel_mma_smooth.h"
//#include "bdd_multi_parallel_mma.h"
//#include "bdd_lbfgs_parallel_mma.h"
//#include "bdd_lbfgs_cuda_mma.h"
//#include "bdd_subgradient.h"
//

#include <variant> 
#include <optional>
#include <nlohmann/json.hpp>

namespace LPMP {

    // glue together various phases of solving:
    // (i) obtain an ILP_input, reorder variables.
    // (ii) preprocess it.
    // (iii) transformation into QBDD format.
    // (iv) give the QBDDs to a specific bdd solver (i.e. mma, parallel_mma, ...).
    // (v) solve the dual.
    // (vi) try to obtain a primal solution.
    
    class bdd_solver {
        public:
            // todo: have using aliases for all solver typenames
            using sequential_mma_float_type = bdd_mma_base<bdd_branch_instruction_bdd_index<float,uint32_t>>;
            using sequential_mma_double_type = bdd_mma_base<bdd_branch_instruction_bdd_index<double, uint32_t>>;
            using parallel_mma_float_type = bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>;
            using parallel_mma_double_type = bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>;
            using lbfgs_parallel_mma_float_type = lbfgs<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>, std::vector<float>, float, std::vector<char>, false>;
            using lbfgs_parallel_mma_double_type = lbfgs<bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>, std::vector<double>, double, std::vector<char>, false>;
            using subgradient_float_type = subgradient<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>, float>;
            using subgradient_double_type = subgradient<bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>, double>;

#ifdef WITH_CUDA
            using cuda_parallel_mma_float_type = bdd_cuda_parallel_mma<float>;
            using cuda_parallel_mma_double_type = bdd_cuda_parallel_mma<double>;
            using cuda_lbfgs_parallel_mma_float_type = lbfgs<bdd_cuda_parallel_mma<float>, thrust::device_vector<float>, float, thrust::device_vector<char>, true>;
            using cuda_lbfgs_parallel_mma_double_type = lbfgs<bdd_cuda_parallel_mma<double>, thrust::device_vector<double>, double, thrust::device_vector<char>, true>;
#endif

            using solver_type = std::variant<
                sequential_mma_float_type, sequential_mma_double_type, parallel_mma_float_type, parallel_mma_double_type, lbfgs_parallel_mma_float_type, lbfgs_parallel_mma_double_type, subgradient_float_type, subgradient_double_type
#ifdef WITH_CUDA
                , cuda_parallel_mma_float_type, cuda_parallel_mma_double_type, cuda_lbfgs_parallel_mma_float_type, cuda_lbfgs_parallel_mma_double_type
#endif
            >;

            bdd_solver() {};
            bdd_solver(const std::string& config);
            bdd_solver(const char* config);
            bdd_solver(const nlohmann::json& config) : config_(config) {};

            static nlohmann::json read_config(const std::string& c);
            virtual ILP_input read_ILP(const nlohmann::json& config);
            virtual void process_ILP(const nlohmann::json& config, ILP_input &ilp);
            static BDD::bdd_collection transform_to_BDDs(const nlohmann::json &config, const ILP_input &ilp);
            virtual solver_type construct_solver(const nlohmann::json& config, const BDD::bdd_collection& bdd_col, const std::vector<double>& costs);
            virtual void solve_dual(const nlohmann::json& config, solver_type& solver);
            virtual std::vector<char> perturbation_rounding(const nlohmann::json& config, solver_type& solver, const ILP_input& ilp);
            static void export_lp(const nlohmann::json& config, const ILP_input& ilp);
            static void export_bdd_graph(const nlohmann::json& config, const BDD::bdd_collection& bdd_col);
            static void export_bdd_lp(const nlohmann::json& config, const BDD::bdd_collection& bdd_col, const std::vector<double>& costs);
            static void print_statistics(const nlohmann::json &config, const ILP_input &ilp, const BDD::bdd_collection &bdd_col);

            void solve(nlohmann::json& config);
            void solve() { solve(config_); };

            double lower_bound();
            void fix_variable(const size_t var, const bool value);
            void fix_variable(const std::string& var, const bool value);
            two_dim_variable_array<std::array<double,2>> min_marginals();
            std::tuple<std::vector<std::string>, std::vector<double>, std::vector<double>> min_marginals_with_variable_names();
            void export_difficult_core();

        private:
            solver_type solver_;
            ILP_input ilp_;
            BDD::bdd_collection bdd_col_;
            nlohmann::json config_;
    };

}
