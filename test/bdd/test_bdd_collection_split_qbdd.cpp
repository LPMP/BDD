#include "bdd_collection/bdd_collection.h"
#include "bdd_solver/bdd_parallel_mma_base.h"
#include "bdd_solver/bdd_branch_instruction.h"
#include <random>
#include "../test.h"
#include "../test_problems.h"
#include "ILP/ILP_input.h"
#include "ILP/ILP_parser.h"
#include "bdd_conversion/bdd_preprocessor.h"

using namespace LPMP;
using namespace BDD;

void run_split_bdd_test(const bool add_implication_bdd)
{
    std::uniform_real_distribution<> d(-10, 10);
    std::mt19937 gen;

    for (size_t i = 2; i < 17; ++i)
    {
        for (size_t k = 1; k < i - 1; ++k)
        {
            for (size_t chunk_size = 2; chunk_size + 1 < i; ++chunk_size)
            {
                bdd_collection bdd_col;
                const size_t card_nr = bdd_col.cardinality_constraint(i, k);
                bdd_parallel_mma_base<bdd_branch_instruction<double,uint16_t>> parallel_mma_card(bdd_col);

                const auto [new_bdd_nrs, new_aux_var] = bdd_col.split_qbdd(card_nr, chunk_size, i, add_implication_bdd);

                test(new_bdd_nrs.size() > 1);
                const size_t nr_chunks = (i + chunk_size - 1) / chunk_size;
                //if (nr_chunks > 2)
                //    test(new_bdd_nrs.size() == nr_chunks + 1); // additional transitive closure implication BDD present
                //else
                //    test(new_bdd_nrs.size() == 2);

                if (k == 1)
                {
                    test(new_aux_var == i + 2 * (nr_chunks - 1));
                }

                bdd_col.remove(card_nr);
                bdd_parallel_mma_base<bdd_branch_instruction<double,uint16_t>> parallel_mma_split_card(bdd_col);

                std::vector<double> costs(i);
                for (size_t v = 0; v < i; ++v)
                    costs[v] = d(gen);
                parallel_mma_card.update_costs({}, costs);
                parallel_mma_split_card.update_costs({}, costs);
                for (size_t iter = 0; iter < 200; ++iter)
                    parallel_mma_split_card.iteration();

                std::sort(costs.begin(), costs.end());
                std::partial_sum(costs.begin(), costs.end(), costs.begin());
                std::cout << "minimum cost = " << costs[k - 1] << " mma on single cardinality constraint lb = " << parallel_mma_card.lower_bound() << ", mma on split cardinality constraint lb = " << parallel_mma_split_card.lower_bound() << "\n";
                test(std::abs(parallel_mma_card.lower_bound() - costs[k - 1]) <= 1e-6);
                test(std::abs(parallel_mma_split_card.lower_bound() - costs[k - 1]) <= 1e-6);
            }
        }
    }

    for(const auto& test_problem : test_problems)
    {
        ILP_input ilp = ILP_parser::parse_string(test_problem);
        bdd_preprocessor preprocessor_long_bdds(ilp);
        auto bdd_col = preprocessor_long_bdds.get_bdd_collection();
        bdd_parallel_mma_base<bdd_branch_instruction<double,uint16_t>> parallel_mma_long_bdds(bdd_col, ilp.objective());
        const size_t nr_orig_bdds = bdd_col.nr_bdds();
        std::vector<size_t> bdds_to_remove;
        size_t aux_var = ilp.nr_variables();
        for (size_t bdd_nr = 0; bdd_nr < nr_orig_bdds; ++bdd_nr)
        {
            const auto [new_bdd_nrs, new_aux_var] = bdd_col.split_qbdd(bdd_nr, 2, aux_var, add_implication_bdd);
            if(new_bdd_nrs.size() > 1)
                bdds_to_remove.push_back(bdd_nr);
            aux_var = new_aux_var;
        }
        bdd_col.remove(bdds_to_remove.begin(), bdds_to_remove.end());
        bdd_parallel_mma_base<bdd_branch_instruction<double,uint16_t>> parallel_mma_split_bdds(bdd_col, ilp.objective());
        for (size_t iter = 0; iter < 300; ++iter)
            parallel_mma_long_bdds.iteration();
        for (size_t iter = 0; iter < 300; ++iter)
            parallel_mma_split_bdds.iteration();

        test(std::abs(parallel_mma_long_bdds.lower_bound() - parallel_mma_split_bdds.lower_bound()) <= 1e-6);
    }
}

int main(int argc, char** argv)
{
    run_split_bdd_test(false);
    run_split_bdd_test(true);
}