#include "bdd_collection/bdd_collection.h"
#include "bdd_parallel_mma.h"
#include <random>
#include "../test.h"

using namespace LPMP;
using namespace BDD;

int main(int argc, char** argv)
{
    std::uniform_real_distribution<> d(-10, 10);
    std::mt19937 gen;

    for (size_t i = 2; i < 17; ++i)
    {
        for(size_t k=1; k<i-1; ++k)
        {
        for (size_t chunk_size = 2; chunk_size + 1 < i; ++chunk_size)
        {
            bdd_collection bdd_col;
            const size_t card_nr = bdd_col.cardinality_constraint(i, k);
            bdd_parallel_mma<double> parallel_mma_card(bdd_col);

            const auto [new_bdd_nrs, new_aux_var] = bdd_col.split_qbdd(card_nr, chunk_size, i);
            test(new_bdd_nrs.size() > 1);
            test(new_bdd_nrs.size() == (i + chunk_size - 1) / chunk_size);
            if (k == 1)
                test(new_aux_var == i + 2 * (new_bdd_nrs.size() - 1));
            bdd_col.remove(card_nr);
            bdd_parallel_mma<double> parallel_mma_split_card(bdd_col);

            std::vector<double> costs(i);
            for (size_t v = 0; v < i; ++v)
                costs[v] = d(gen);
            parallel_mma_card.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            parallel_mma_split_card.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            for (size_t iter = 0; iter < 100; ++iter)
                parallel_mma_split_card.iteration();

            std::sort(costs.begin(), costs.end());
            std::partial_sum(costs.begin(), costs.end(), costs.begin());
            std::cout << "minimum cost = " << costs[k-1] << " mma on single cardinality constraint lb = " << parallel_mma_card.lower_bound() << ", mma on split cardinality constraint lb = " << parallel_mma_split_card.lower_bound() << "\n";
            test(std::abs(parallel_mma_card.lower_bound() - costs[k-1]) <= 1e-6);
            test(std::abs(parallel_mma_split_card.lower_bound() - costs[k-1]) <= 1e-6);
        }
        }
    }
}