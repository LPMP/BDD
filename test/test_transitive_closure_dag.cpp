#include "transitive_closure_dag.h"
#include "test.h"
#include <random>
#include <unordered_set>
#include <iostream>
#include <algorithm>

using namespace LPMP;

std::mt19937 gen;

std::tuple<interval_rep, std::vector<size_t>> generate_random_interval(const size_t n, double p)
{
    std::vector<size_t> elems;
    std::bernoulli_distribution d(p);
    for(size_t i=0; i<n; ++i)
        if (d(gen))
            elems.push_back(i);

    return {interval_rep(elems.begin(), elems.end()), elems};
}

std::vector<std::array<size_t,2>> generate_random_dag(const size_t nr_nodes, const double p)
{
    std::vector<std::array<size_t,2>> arcs;
    std::bernoulli_distribution d(p);
    for(size_t i=0; i<nr_nodes; ++i)
        for(size_t j=i+1; j<nr_nodes; ++j)
            if (d(gen))
                arcs.push_back({i, j});

    std::vector<size_t> permutation(nr_nodes);
    for(size_t i=0; i<nr_nodes; ++i)
        permutation[i] = i;
    std::shuffle(permutation.begin(), permutation.end(), gen);
    for(auto& [i,j] : arcs)
    {
        i = permutation[i];
        j = permutation[j];
    }

    return arcs;
}

std::vector<char> reachability(const size_t i, const std::vector<std::array<size_t,2>>& arcs)
{
    size_t nr_nodes = 0;
    for(const auto& [i,j] : arcs)
        nr_nodes = std::max(nr_nodes, std::max(i,j) + 1);
    assert(i < nr_nodes);
    std::vector<char> reachable(nr_nodes, 0);
    reachable[i] = 1;

    for (size_t round = 0; round < nr_nodes; ++round)
        for (const auto [v, w] : arcs)
            if (reachable[v])
                reachable[w] = 1;

    return reachable;
}

int main(int argc, char** argv)
{
    {
        std::vector<size_t> S = {1, 2, 4, 6, 7, 8};
        interval_rep intr(S.begin(), S.end());
        for (const size_t i : S)
            test(intr[i]);
        test(!intr[0]);
        test(!intr[3]);
        test(!intr[5]);
        test(!intr[9]);
    }

    {
        std::vector<std::array<size_t, 2>> dag = {
            {2, 1}, {3, 1}, {0, 2}, {0, 3}};

        transitive_closure tc(dag);
        for (const auto [i, j] : dag)
            test(tc(i, j));
        test(tc(0, 1));
        test(!tc(2, 3));
        test(!tc(3, 2));
    }

    // random interval inclusion test
    for(size_t n=2; n<100; ++n)
        for(double p=0.0; p<=1.0; p += 0.1)
        {
            auto [intr, elems] = generate_random_interval(n, p);
            std::unordered_set<size_t> elems_hash(elems.begin(), elems.end());
            for(size_t i=0; i<n; ++i)
                if (elems_hash.count(i) > 0)
                    test(intr[i] == true);
                else
                    test(intr[i] == false);
        }

    // random interval merging test
    for (size_t n1 = 2; n1 < 20; ++n1)
        for (double p1 = 0.0; p1 <= 1.0; p1 += 0.1)
            for (size_t n2 = 2; n2 < 20; ++n2)
                for (double p2 = 0.0; p2 <= 1.0; p2 += 0.1)
                {
                    auto [intr1, elems1] = generate_random_interval(n1, p1);
                    std::unordered_set<size_t> elems_hash1(elems1.begin(), elems1.end());
                    auto [intr2, elems2] = generate_random_interval(n2, p2);
                    std::unordered_set<size_t> elems_hash2(elems2.begin(), elems2.end());
                    const interval_rep u = merge(intr1, intr2);
                    //std::cout << intr1 << " merge " << intr2 << " = " << u << "\n";
                    for(size_t i=0; i<std::max(n1,n2)+1; ++i)
                        if (elems_hash1.count(i) > 0 || elems_hash2.count(i) > 0)
                            test(u[i] == true);
                        else
                            test(u[i] == false);
                }

    // random dag reachability test
    for (size_t n = 2; n < 20; ++n)
    {
        for (double p = 0.1; p <= 1.0; p += 0.1)
        {
            const auto arcs = generate_random_dag(n, p);
            if(arcs.size() == 0)
                continue;

            transitive_closure tc(arcs);
            for (size_t i = 0; i < tc.nr_nodes(); ++i)
            {
                const auto reachable = reachability(i, arcs);
                for (size_t j = 0; j < tc.nr_nodes(); ++j)
                    test(tc(i, j) == reachable[j]);
            }
        }
    }
}