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
        for (size_t chunk_size = 2; chunk_size + 1 < i; ++chunk_size)
        {
            bdd_collection bdd_col;
            const size_t simplex_nr = bdd_col.simplex_constraint(i);
            bdd_parallel_mma<double> parallel_mma_simplex(bdd_col);

            const auto [new_bdd_nrs, new_aux_var] = bdd_col.split_qbdd(simplex_nr, chunk_size, i);
            test(new_bdd_nrs.size() > 1);
            test(new_bdd_nrs.size() == (i + chunk_size - 1) / chunk_size);
            test(new_aux_var == i + 2*(new_bdd_nrs.size()-1));
            bdd_col.remove(simplex_nr);
            bdd_parallel_mma<double> parallel_mma_split_simplex(bdd_col);

            std::vector<double> costs(i);
            for (size_t v = 0; v < i; ++v)
                costs[v] = d(gen);
            parallel_mma_simplex.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            parallel_mma_split_simplex.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            for (size_t iter = 0; iter < 100; ++iter)
                parallel_mma_split_simplex.iteration();

            std::cout << "cost min = " << *std::min_element(costs.begin(), costs.end()) << " mma on single simplex lb = " << parallel_mma_simplex.lower_bound() << ", mma on split simplices lb = " << parallel_mma_split_simplex.lower_bound() << "\n";
            test(std::abs(parallel_mma_simplex.lower_bound() - *std::min_element(costs.begin(), costs.end())) <= 1e-6);
            test(std::abs(parallel_mma_split_simplex.lower_bound() - *std::min_element(costs.begin(), costs.end())) <= 1e-6);
        }
    }
}