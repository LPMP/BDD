#include "bdd_preprocessor.h"
#include <iostream>
#include <chrono>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <cmath>
#include "time_measure_util.h"
#include <omp.h>

namespace LPMP {

    bdd_preprocessor::bdd_preprocessor(const ILP_input& input, const bool constraint_groups)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(bdd_collection.nr_bdds() == 0);
        // first transform linear inequalities into BDDs
        std::cout << "[bdd_preprocessor] convert " << input.constraints().size() << " linear inequalities.\n";

#ifdef _OPENMP
        const size_t nr_threads = omp_get_num_threads();
#else
        const size_t nr_threads = 1;
#endif
        std::cout << "[bdd preprocessor] #threads = " << nr_threads << "\n";

#pragma omp parallel for ordered schedule(static) num_threads(nr_threads)
        for(size_t tid=0; tid<nr_threads; ++tid)
        {
            std::vector<int> coefficients;
            std::vector<std::size_t> variables;
            BDD::bdd_mgr bdd_mgr;
            bdd_converter converter(bdd_mgr);
            BDD::bdd_collection cur_bdd_collection;

            const size_t first_constr = input.constraints().size()/nr_threads * tid;
            const size_t last_constr = (tid+1 == nr_threads) ? input.constraints().size() : (input.constraints().size()/nr_threads) * (tid+1);
            if(tid + 1 == nr_threads)
                assert(last_constr == input.constraints().size());

            for(size_t c=first_constr; c<last_constr; ++c)
            {
                const auto& constraint = input.constraints()[c];
                coefficients.clear();
                variables.clear();
                for(const auto e : constraint.variables) {
                    coefficients.push_back(e.coefficient);
                    variables.push_back(e.var);
                }
                BDD::node_ref bdd = converter.convert_to_bdd(coefficients, constraint.ineq, constraint.right_hand_side);
                if(bdd.is_topsink())
                {
                    if(constraint_groups == true && input.nr_constraint_groups() > 0)
                        throw std::runtime_error("constraint groups and empty constraints not both supported");
                    continue;
                }
                else if(bdd.is_botsink())
                    throw std::runtime_error("problem is infeasible");
                const size_t bdd_nr = cur_bdd_collection.add_bdd(bdd);
                cur_bdd_collection.reorder(bdd_nr);
                assert(cur_bdd_collection.is_reordered(bdd_nr));
                cur_bdd_collection.rebase(bdd_nr, variables.begin(), variables.end());
            }
#pragma omp ordered
            {
                bdd_collection.append(cur_bdd_collection);
            }
        }

        // need not hold if some constraints are empty
        //assert(bdd_collection.nr_bdds() == input.constraints().size());

        if(constraint_groups == true) // coalesce BDDs 
        {
            std::cout << "[bdd_preprocessor] form " << input.nr_constraint_groups() << " constraint groups.\n";
            std::vector<size_t> bdd_nrs;
            for(size_t c=0; c<input.nr_constraint_groups(); ++c)
            {
                auto [c_begin, c_end] = input.constraint_group(c);
                const size_t coalesced_bdd_nr = bdd_collection.bdd_and(c_begin, c_end);
                bdd_collection.reorder(coalesced_bdd_nr);
                assert(bdd_collection.is_reordered(coalesced_bdd_nr));
            }

            // remove BDDs that were coalesced
            std::vector<size_t> unused_bdd_nrs;
            for(size_t c=0; c<input.nr_constraint_groups(); ++c)
            {
                auto [c_begin, c_end] = input.constraint_group(c);
                unused_bdd_nrs.insert(unused_bdd_nrs.end(), c_begin, c_end);
            }
            std::sort(unused_bdd_nrs.begin(), unused_bdd_nrs.end());
            auto new_unused_bdd_nrs_end = std::unique(unused_bdd_nrs.begin(), unused_bdd_nrs.end());
            unused_bdd_nrs.resize(std::distance(unused_bdd_nrs.begin(), new_unused_bdd_nrs_end));
            std::cout << "[bdd_preprocessor] remove " << unused_bdd_nrs.size() << " original BDDs.\n";

            //std::fstream fs;
            //fs.open ("kwas.dot", std::fstream::in | std::fstream::out | std::ofstream::trunc);
            //bdd_collection.export_graphviz(coalesced_bdd_nr, fs);
            //add_bdd(bdd_collection[coalesced_bdd_nr]);
            bdd_collection.remove(unused_bdd_nrs.begin(), unused_bdd_nrs.end());
        }

        // transform all BDDs to qbdds
        std::cout << "[bdd preprocessor] Transform BDDs into QBDDs\n";
        std::vector<size_t> bdds_to_remove;
        const size_t orig_nr_bdds = bdd_collection.nr_bdds();
#pragma omp parallel for schedule(static)
        for(size_t bdd_nr = 0; bdd_nr<orig_nr_bdds; ++bdd_nr)
        {
            if(!bdd_collection.is_qbdd(bdd_nr))
            {
#pragma omp critical
                {
                    bdd_collection.make_qbdd(bdd_nr);
                    bdds_to_remove.push_back(bdd_nr);
                }
            }
        }
        std::sort(bdds_to_remove.begin(), bdds_to_remove.end());
        std::cout << "[bdd preprocessor] " << bdds_to_remove.size() << " BDDs had to be transformed\n";
        bdd_collection.remove(bdds_to_remove.begin(), bdds_to_remove.end());

        // second, preprocess BDDs, TODO: do this separately!
        /*
        if(preprocessing_arg.getValue().size() > 0)
        {
            for(const std::string& preprocessing : preprocessing_arg.getValue())
            {
                if(preprocessing == "bridge")
                    bdd_pre.set_coalesce_bridge();
                else if(preprocessing == "subsumption")
                    bdd_pre.set_coalesce_subsumption();
                else if(preprocessing == "contiguous_overlap")
                    bdd_pre.set_coalesce_contiguous_overlap();
                else if(preprocessing == "subsumption_except_one")
                    bdd_pre.set_coalesce_subsumption_except_one();
                else if(preprocessing == "partial_contiguous_overlap")
                    bdd_pre.set_coalesce_partial_contiguous_overlap();
                else if(preprocessing == "cliques")
                    bdd_pre.set_coalesce_cliques();
                else
                    throw std::runtime_error("bdd preprocessing argument " + preprocessing + " not recognized.");
            }
            bdd_pre.coalesce_bdd_collection();

            for(size_t bdd_nr=0; bdd_nr<bdd_pre.get_bdd_collection().nr_bdds(); ++bdd_nr)
                add_bdd(bdd_pre.get_bdd_collection()[bdd_nr]);
        }
        */
    }

    void bdd_preprocessor::add_bdd(BDD::node_ref bdd)
    {
        bdds.push_back(bdd);
        std::vector<size_t> empty;
        indices.push_back(empty.begin(), empty.end());
        nr_variables = std::max(nr_variables, bdd.variables().back()+1);
    }

    void bdd_preprocessor::add_bdd(BDD::bdd_collection_entry bdd)
    {
        //throw std::runtime_error("bdd_add in bdd_preprocessor not implemted");
    }

    void bdd_preprocessor::construct_bdd_collection()
    {
        return; // TODO: remove
        assert(bdd_collection.nr_bdds() == 0);
        for(size_t i=0; i<bdds.size(); ++i)
        {
            const size_t bdd_nr = bdd_collection.add_bdd(bdds[i]);
            if(indices[i].size() > 0)
            {
                bdd_collection.rebase(bdd_nr, indices[i].begin(), indices[i].end());
            } 
        }
        std::cout << "initialized bdd collection\n";
        assert(bdd_collection.nr_bdds() == bdds.size()); 
    }

    void bdd_preprocessor::rebase_bdds()
    {
        assert(bdd_collection.nr_bdds() == 0);
        for(size_t i=0; i<bdds.size(); ++i)
        {
            const size_t bdd_nr = bdd_collection.add_bdd(bdds[i]);
            if(indices[i].size() > 0)
            {
                bdd_collection.rebase(bdd_nr, indices[i].begin(), indices[i].end());
            } 
        }
        std::cout << "initialized bdd collection\n";
        assert(bdd_collection.nr_bdds() == bdds.size());
        //bdd_mgr.collect_garbage();

        /*
        for(size_t i=0; i<bdds.size(); ++i)
        {
            if(indices[i].size() > 0)
            {
                assert(indices[i].size() == bdds[i].variables().size());
                bdds[i] = bdd_mgr.rebase(bdds[i], indices[i].begin(), indices[i].end()); 
            } 
        }
        std::cout << "rebased bdd base\n";

        indices.clear();
        for(size_t i=0; i<bdds.size(); ++i)
        {
            std::vector<size_t> empty;
            indices.push_back(empty.begin(), empty.end());
        }
        */

    }

    template<typename ITERATOR_1, typename ITERATOR_2>
    bool is_prefix(ITERATOR_1 v1_begin, ITERATOR_1 v1_end, ITERATOR_2 v2_begin, ITERATOR_2 v2_end)
    {
        assert(std::is_sorted(v1_begin, v1_end));
        assert(std::unique(v1_begin, v1_end) == v1_end);
        assert(std::is_sorted(v2_begin, v2_end));
        //assert(std::unique(v2_begin, v2_end) == v2_end);
        assert(std::distance(v1_begin, v1_end) <= std::distance(v2_begin, v2_end));

        for(; v1_begin!=v1_end; v1_begin++, v2_begin++)
            if(*v1_begin != *v2_begin)
                return false;
        return true;
    }

    template<typename ITERATOR_1, typename ITERATOR_2>
    bool is_suffix(ITERATOR_1 v1_begin, ITERATOR_1 v1_end, ITERATOR_2 v2_begin, ITERATOR_2 v2_end)
    {
        assert(std::is_sorted(v1_begin, v1_end));
        assert(std::unique(v1_begin, v1_end) == v1_end);
        assert(std::is_sorted(v2_begin, v2_end));
        //assert(std::unique(v2_begin, v2_end) == v2_end);
        assert(std::distance(v1_begin, v1_end) <= std::distance(v2_begin, v2_end));

        const size_t v1_size = std::distance(v1_begin, v1_end);
        v2_begin = v2_end - v1_size;
        for(; v1_begin!=v1_end; v1_begin++, v2_begin++)
            if(*v1_begin != *v2_begin)
                return false;
        return true;
    }

    std::vector<bdd_preprocessor::coalesce_candidate> bdd_preprocessor::bridge_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const
    {
        // find bdd pairs such that there exist a variable that is only covered by these two
        std::vector<coalesce_candidate> candidates;
        //std::unordered_set<std::array<size_t,2>> considered_candidates;
        tsl::robin_set<std::array<size_t,2>> considered_candidates;

        for(size_t v=0; v<var_bdd_adjacency.size(); ++v)
        {
            if(var_bdd_adjacency[v].size() == 2)
            {
                const size_t bdd_1 = std::min(var_bdd_adjacency(v,0), var_bdd_adjacency(v,1));
                const size_t bdd_2 = std::max(var_bdd_adjacency(v,0), var_bdd_adjacency(v,1));
                if(considered_candidates.count({bdd_1, bdd_2}) > 0)
                    continue;
                considered_candidates.insert({bdd_1, bdd_2});
                candidates.push_back({bdd_1,bdd_2});
            }
        }

        return candidates;
    }

        std::vector<bdd_preprocessor::coalesce_candidate> bdd_preprocessor::compute_candidates(const bool subsumption, const bool contiguous_overlap, const bool subsumption_except_one, const bool partial_contiguous_overlap, const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const

    {
        assert(subsumption || contiguous_overlap || subsumption_except_one || partial_contiguous_overlap);
        std::vector<coalesce_candidate> candidates;
        //std::unordered_set<std::array<size_t,2>> considered_candidates;
        tsl::robin_set<std::array<size_t,2>> considered_candidates;
        std::vector<size_t> overlap;

        for(size_t v=0; v<var_bdd_adjacency.size(); ++v)
        {
            for(size_t bdd_idx_1=0; bdd_idx_1<var_bdd_adjacency[v].size(); ++bdd_idx_1)
            {
                const size_t bdd_1 = var_bdd_adjacency(v,bdd_idx_1);
                for(size_t bdd_idx_2=bdd_idx_1+1; bdd_idx_2<var_bdd_adjacency[v].size(); ++bdd_idx_2)
                {
                    const size_t bdd_2 = var_bdd_adjacency(v,bdd_idx_2);
                    if(considered_candidates.count({std::min(bdd_1,bdd_2), std::max(bdd_1,bdd_2)}) > 0)
                        continue;
                    //std::cout << "examining coalesce candidate " << bdd_1 << ", " << bdd_2 << "\n";
                    considered_candidates.insert({std::min(bdd_1,bdd_2), std::max(bdd_1, bdd_2)});

                    overlap.clear();
                    std::set_intersection(
                            bdd_var_adjacency[bdd_1].begin(), bdd_var_adjacency[bdd_1].end(), 
                            bdd_var_adjacency[bdd_2].begin(), bdd_var_adjacency[bdd_2].end(), 
                            std::back_inserter(overlap)
                            );
                    assert(overlap.size() > 0);

                    if(subsumption)
                    {
                        // check if variables of one bdd is contained in the variables of the other (or has at most one variable that is not contained in the other
                        const bool subsumption_present = [&]() {
                            if(bdd_var_adjacency[bdd_1].size() == overlap.size())
                                return true;
                            if(bdd_var_adjacency[bdd_2].size() == overlap.size())
                                return true;
                            return false;
                        }();

                        if(subsumption_present)
                        {
                            candidates.push_back({std::min(bdd_1,bdd_2), std::max(bdd_1,bdd_2)});
                            continue;
                        }
                    }

                    if(subsumption_except_one)
                    {
                        const bool subsumption_except_one_present = [&]() {
                            if(bdd_var_adjacency[bdd_1].size() == overlap.size()+1)
                            {
                                // check if either first 
                                if(bdd_var_adjacency(bdd_1,0) != overlap[0])
                                    return true;
                                // or last variabel of bdd 1 is not in overlap
                                if(bdd_var_adjacency[bdd_1].back() != overlap.back())
                                    return true;
                            }

                            if(bdd_var_adjacency[bdd_2].size() == overlap.size()+1)
                            {
                                if(bdd_var_adjacency(bdd_2,0) != overlap[0])
                                    return true;
                                if(bdd_var_adjacency[bdd_2].back() != overlap.back())
                                    return true;
                            }

                            return false;
                        }();

                        if(subsumption_except_one_present)
                        {
                            candidates.push_back({std::min(bdd_1,bdd_2), std::max(bdd_1,bdd_2)});
                            continue;
                        }
                    }

                    if(contiguous_overlap)
                    {
                        // check if overlapping variables range is an (approximate) suffix or prefix of either bdd
                        const bool contiguous_overlap_present = [&]() {
                            if(is_suffix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_1].begin(), bdd_var_adjacency[bdd_1].end())
                                    && is_prefix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_2].begin(), bdd_var_adjacency[bdd_2].end()))
                                return true; 
                            if(is_suffix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_2].begin(), bdd_var_adjacency[bdd_2].end())
                                    && is_prefix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_1].begin(), bdd_var_adjacency[bdd_1].end()))
                                return true; 
                            return false;
                        }();

                        if(contiguous_overlap_present)
                        {
                            candidates.push_back({std::min(bdd_1,bdd_2), std::max(bdd_1,bdd_2)});
                            continue;
                        }
                    }

                    if(partial_contiguous_overlap)
                    {
                        const bool partial_contiguous_overlap_present = [&]() {
                            if(is_suffix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_1].begin(), bdd_var_adjacency[bdd_1].end()))
                                return true;
                            if(is_prefix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_2].begin(), bdd_var_adjacency[bdd_2].end()))
                                return true; 
                            if(is_suffix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_2].begin(), bdd_var_adjacency[bdd_2].end()))
                                return true;
                            if(is_prefix(overlap.begin(), overlap.end(), bdd_var_adjacency[bdd_1].begin(), bdd_var_adjacency[bdd_1].end()))
                                return true; 
                            return false;
                        }();

                        if(partial_contiguous_overlap_present)
                        {
                            candidates.push_back({std::min(bdd_1,bdd_2), std::max(bdd_1,bdd_2)});
                            continue;
                        }
                    }
                }
            }
        }

        return candidates;
    }

    void bdd_preprocessor::coalesce()
    {
        std::cout << "rebasing bdds\n";
        rebase_bdds(); 

        for(size_t coalesce_round = 1; coalesce_round<10; ++coalesce_round) // iterate until no coalescing takes place
        {
            std::cout << "coalescing bdds, round = " << coalesce_round << ", nr bdds = " << bdd_collection.nr_bdds() << "\n"; 
            // TODO: sort bdds according to some criterion, e.g. largest/smallest nr of variables covered, largest/smallest nr of nodes.

            // iterate over bdds in order of ascending size
            std::vector<size_t> bdd_coalesce_order(bdd_collection.nr_bdds());
            std::iota(bdd_coalesce_order.begin(), bdd_coalesce_order.end(), 0);
            std::sort(bdd_coalesce_order.begin(), bdd_coalesce_order.end(),
                    [&](const size_t i, const size_t j) {
                    return bdd_collection.nr_bdd_nodes(i) < bdd_collection.nr_bdd_nodes(j);
                    });

            std::vector<size_t> bdd_indices(bdd_collection.nr_bdds());
            std::iota(bdd_indices.begin(), bdd_indices.end(), 0);

            two_dim_variable_array<size_t> bdd_var_adjacency;
            std::vector<size_t> var_bdd_adjacency_size(nr_variables, 0);
            // first compute variable bdd incidence matrix
            for(size_t j=0; j<bdd_collection.nr_bdds(); ++j)
            {
                const auto vars = bdd_collection.variables(j);
                bdd_var_adjacency.push_back(vars.begin(), vars.end());
                for(const size_t v : vars)
                {
                    assert(v < var_bdd_adjacency_size.size());
                    var_bdd_adjacency_size[v]++;
                }
            }

            two_dim_variable_array<size_t> var_bdd_adjacency(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end());
            std::fill(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end(), 0);
            for(size_t j=0; j<bdd_var_adjacency.size(); ++j)
                for(const size_t v : bdd_var_adjacency[j])
                    var_bdd_adjacency(v, var_bdd_adjacency_size[v]++) = j;

            std::vector<char> active_bdds(bdd_collection.nr_bdds(), true);

            // for each bdd, search for overlapping ones. Sort by greatest overlap and test if intersection grows moderately only. If so, keep intersection and remove individual bdds.
            //std::unordered_map<size_t,size_t> checked_bdd_pairs; // value is timestamp
            tsl::robin_map<size_t,size_t> checked_bdd_pairs; // value is timestamp
            //std::unordered_map<size_t,size_t> adjacent_bdds;
            tsl::robin_map<size_t,size_t> adjacent_bdds;
            struct bdd_intersection { size_t bdd; size_t nr_common_vars; };
            std::vector<bdd_intersection> adjacent_bdds_sorted;

            for(size_t j_idx=0; j_idx<bdd_var_adjacency.size(); ++j_idx)
            {
                const size_t j = bdd_coalesce_order[j_idx];
                if(active_bdds[j] == false)
                    continue;

                checked_bdd_pairs.clear();
                adjacent_bdds.clear();

                for(size_t i : bdd_var_adjacency[j])
                    for(size_t jj : var_bdd_adjacency[i])
                        if(j != jj && active_bdds[jj] && !(checked_bdd_pairs.count(jj) > 0))
                            adjacent_bdds[jj]++;

                // sort by largest intersection
                adjacent_bdds_sorted.clear();
                adjacent_bdds_sorted.reserve(adjacent_bdds.size());
                for(const auto [bdd_nr, nr_common_vars] : adjacent_bdds)
                    adjacent_bdds_sorted.push_back({bdd_nr, nr_common_vars});
                std::sort(adjacent_bdds_sorted.begin(), adjacent_bdds_sorted.end(), 
                        [](const bdd_intersection a, const bdd_intersection b) { return a.nr_common_vars > b.nr_common_vars; });

                // go over sorted bdd list and try out intersection. If growth moderate, replace bdd nr j by intersection and remove the other bdd.
                for(const auto [jj, nr_common_vars] : adjacent_bdds_sorted)
                {
                    checked_bdd_pairs.insert({jj,0});
                    const size_t node_limit = 
                        2*std::min(bdd_collection.nr_bdd_nodes(bdd_indices[j]), bdd_collection.nr_bdd_nodes(bdd_indices[jj])) +
                        std::max(bdd_collection.nr_bdd_nodes(bdd_indices[j]), bdd_collection.nr_bdd_nodes(bdd_indices[jj]));
                    //const size_t node_limit = 1.5*(bdd_collection.nr_bdd_nodes(bdd_indices[j]) + bdd_collection.nr_bdd_nodes(bdd_indices[jj]));
                    const size_t intersect = bdd_collection.bdd_and(bdd_indices[j], bdd_indices[jj], node_limit);
                    if(intersect != std::numeric_limits<size_t>::max())
                    {
                        //std::cout << "nr nodes before = " << bdds[j].nr_nodes() << ", " << bdds[jj].nr_nodes() << ", nodes after = " << intersect.nr_nodes() << " = " <<  (bdds[j] & bdds[jj]).nr_nodes() << "\n";
                        active_bdds[j] = false;
                        active_bdds[jj] = false;
                        assert(intersect == active_bdds.size());
                        assert(intersect +1 == bdd_collection.nr_bdds());
                        active_bdds.push_back(true);
                        bdd_indices[j] = intersect;
                    } 
                }
            }

            std::vector<size_t> bdds_to_remove;
            for(size_t i=0; i<active_bdds.size(); ++i)
                if(active_bdds[i] == false)
                    bdds_to_remove.push_back(i);
            if(bdds_to_remove.size() == 0)
                break;
            bdd_collection.remove(bdds_to_remove.begin(), bdds_to_remove.end());

            assert(bdd_collection.nr_bdds() < bdd_indices.size());
        }
        /*
        for(;;) // iterate until no coalescing takes place
        { 
            std::cout << "coalescing bdds, nr bdds = " << bdds.size() << "\n"; 
            // TODO: sort bdds according to some criterion, e.g. largest/smallest nr of variables covered, largest/smallest nr of nodes.

            two_dim_variable_array<size_t> bdd_var_adjacency;
            std::vector<size_t> var_bdd_adjacency_size(bdd_mgr.nr_variables(), 0);
            // first compute variable bdd incidence matrix
            for(size_t j=0; j<bdds.size(); ++j)
            {
                const auto vars = bdds[j].variables();
                bdd_var_adjacency.push_back(vars.begin(), vars.end());
                for(const size_t v : vars)
                    var_bdd_adjacency_size[v]++;
            }

            two_dim_variable_array<size_t> var_bdd_adjacency(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end());
            std::fill(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end(), 0);
            for(size_t j=0; j<bdd_var_adjacency.size(); ++j)
                for(const size_t v : bdd_var_adjacency[j])
                    var_bdd_adjacency(v, var_bdd_adjacency_size[v]++) = j;

            std::vector<char> active_bdds(bdds.size(), true);

            // for each bdd, search for overlapping ones. Sort by greatest overlap and test if intersection grows moderately only. If so, keep intersection and remove individual bdds.
            std::unordered_map<size_t,size_t> checked_bdd_pairs; // value is timestamp
            std::unordered_map<size_t,size_t> adjacent_bdds;
            struct bdd_intersection { size_t bdd; size_t nr_common_vars; };
            std::vector<bdd_intersection> adjacent_bdds_sorted;

            for(size_t j=0; j<bdds.size(); ++j)
            {
                if(active_bdds[j] == false)
                    continue;

                checked_bdd_pairs.clear();
                adjacent_bdds.clear();
                size_t cur_nr_nodes = bdds[j].nr_nodes();

                for(size_t i : bdd_var_adjacency[j])
                    for(size_t jj : var_bdd_adjacency[i])
                        if(j != jj && active_bdds[jj] && !(checked_bdd_pairs.count(jj) > 0))
                            adjacent_bdds[jj]++;

                // sort by largest intersection
                adjacent_bdds_sorted.clear();
                adjacent_bdds_sorted.reserve(adjacent_bdds.size());
                for(const auto [bdd_nr, nr_common_vars] : adjacent_bdds)
                    adjacent_bdds_sorted.push_back({bdd_nr, nr_common_vars});
                std::sort(adjacent_bdds_sorted.begin(), adjacent_bdds_sorted.end(), 
                        [](const bdd_intersection a, const bdd_intersection b) { return a.nr_common_vars > b.nr_common_vars; });

                // go over sorted bdd list and try out intersection. If growth moderate, replace bdd nr j by intersection and remove the other bdd.
                for(const auto [jj, nr_common_vars] : adjacent_bdds_sorted)
                {
                    checked_bdd_pairs.insert({jj,0});
                    const size_t jj_nr_nodes = bdds[jj].nr_nodes();
                    const size_t node_limit = 3*(cur_nr_nodes + jj_nr_nodes);
                    auto [intersect, nr_intersect_nodes] = bdd_mgr.and_rec_limited(bdds[j], bdds[jj], cur_nr_nodes * jj_nr_nodes);
                    if(intersect.address() != nullptr && intersect.nr_nodes() <= node_limit)
                    {
                        //std::cout << "nr nodes before = " << bdds[j].nr_nodes() << ", " << bdds[jj].nr_nodes() << ", nodes after = " << intersect.nr_nodes() << " = " <<  (bdds[j] & bdds[jj]).nr_nodes() << "\n";
                        bdds[j] = intersect;
                        active_bdds[jj] = false;
                        cur_nr_nodes = intersect.nr_nodes();
                    } 
                }
            }

            // TODO: remove inactive bdds for new round of coalescing until no further progress can be made
            std::vector<BDD::node_ref> new_bdds;
            for(size_t j=0; j<bdds.size(); ++j)
                if(active_bdds[j])
                    new_bdds.push_back(bdds[j]);

            std::swap(bdds, new_bdds);
            if(bdds.size() == new_bdds.size())
                break;
        }
    */

        //bdd_mgr.collect_garbage();
    }

    void bdd_preprocessor::coalesce_bdd_collection()
    {
        construct_bdd_collection();

        //std::unordered_set<std::array<size_t,2>> intersected_bdds;
        tsl::robin_set<std::array<size_t,2>> intersected_bdds;

        if(coalesce_cliques_)
        {
            std::cout << "coalesce cliques\n";
            throw std::runtime_error("disabled currently.");
            //coalesce_cliques();
        }

        for(size_t coalesce_round=1;; ++coalesce_round)
        {
            struct bdd_intersection {
                std::array<size_t,2> old_bdd;
                size_t new_bdd;
            };
            std::vector<bdd_intersection> current_bdd_intersections;

            std::cout << "coalesce round " << coalesce_round << "\n";
            two_dim_variable_array<size_t> bdd_var_adj, var_bdd_adj;
            std::tie(bdd_var_adj, var_bdd_adj) = construct_bdd_var_adjacency(bdd_collection);

            auto filter_out_candidates = [&](std::vector<coalesce_candidate>& candidates) {
                std::cout << "before filtering: " << candidates.size() << "\n";
                std::cout << "nr stored intersected bdds = " << intersected_bdds.size() << "\n";
                auto end = std::remove_if(candidates.begin(), candidates.end(), [&](const coalesce_candidate c) {
                        assert(c[0] < c[1]);
                        return intersected_bdds.count({c[0], c[1]}) > 0;
                        });
                candidates.erase(end, candidates.end());
                std::cout << "after filtering: " << candidates.size() << "\n";
            };

            const auto intersection_candidates = [&]() -> std::vector<coalesce_candidate> {
                std::vector<coalesce_candidate> candidates;
                if(this->coalesce_bridge_ == true)
                {
                    auto bridge_candidates = this->bridge_candidates(bdd_var_adj, var_bdd_adj);
                    filter_out_candidates(bridge_candidates);
                    candidates.insert(candidates.end(), bridge_candidates.begin(), bridge_candidates.end());
                    if(candidates.size() * 20 >= bdd_var_adj.size())
                        return candidates;
                }
                if(coalesce_subsumption_ == true)
                {
                    auto subsumption_candidates = this->subsumption_candidates(bdd_var_adj, var_bdd_adj);
                    filter_out_candidates(subsumption_candidates);
                    candidates.insert(candidates.end(), subsumption_candidates.begin(), subsumption_candidates.end());
                    if(candidates.size() * 20 >= bdd_var_adj.size())
                        goto return_candidates;
                }
                if(coalesce_contiguous_overlap_ == true)
                {
                    auto contiguous_overlap_candidates = this->contiguous_overlap_candidates(bdd_var_adj, var_bdd_adj);
                    filter_out_candidates(contiguous_overlap_candidates);
                    candidates.insert(candidates.end(), contiguous_overlap_candidates.begin(), contiguous_overlap_candidates.end());
                    if(candidates.size() * 20 >= bdd_var_adj.size())
                        goto return_candidates;
                }
                if(coalesce_subsumption_except_one_ == true)
                {
                    auto subsumption_candidates = this->subsumption_candidates(bdd_var_adj, var_bdd_adj);
                    filter_out_candidates(subsumption_candidates);
                    candidates.insert(candidates.end(), subsumption_candidates.begin(), subsumption_candidates.end());
                    if(candidates.size() * 20 >= bdd_var_adj.size())
                        goto return_candidates;
                }
                if(coalesce_partial_contiguous_overlap_ == true)
                {
                    auto contiguous_overlap_candidates = this->contiguous_overlap_candidates(bdd_var_adj, var_bdd_adj);
                    filter_out_candidates(contiguous_overlap_candidates);
                    candidates.insert(candidates.end(), contiguous_overlap_candidates.begin(), contiguous_overlap_candidates.end());
                    if(candidates.size() * 20 >= bdd_var_adj.size())
                        goto return_candidates;
                }

return_candidates:
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());
                return candidates;
            }();

            std::cout << "initial number of coalescing candidates = " << intersection_candidates.size() << "\n";

            std::vector<size_t> bdd_indices(bdd_collection.nr_bdds());
            std::iota(bdd_indices.begin(), bdd_indices.end(), 0);

            std::vector<bool> active_bdds(bdd_collection.nr_bdds(), true);

            std::chrono::steady_clock::time_point coalesce_begin = std::chrono::steady_clock::now();
            std::size_t nr_intersections = 0;
            for(const auto candidate : intersection_candidates)
            {
                const size_t bdd_1 = candidate[0];
                const size_t bdd_2 = candidate[1];
                assert(bdd_1 < bdd_2);
                if(intersected_bdds.count({bdd_1, bdd_2}) > 0)
                    continue;
                const size_t node_limit = 
                    2*std::min(bdd_collection.nr_bdd_nodes(bdd_1), bdd_collection.nr_bdd_nodes(bdd_2)) +
                    std::max(bdd_collection.nr_bdd_nodes(bdd_1), bdd_collection.nr_bdd_nodes(bdd_2));
                //const size_t node_limit = 1.5*(bdd_collection.nr_bdd_nodes(bdd_indices[j]) + bdd_collection.nr_bdd_nodes(bdd_indices[jj]));
                const size_t intersect = bdd_collection.bdd_and(bdd_1, bdd_2, node_limit);
                intersected_bdds.insert({bdd_1, bdd_2});
                if(intersect != std::numeric_limits<size_t>::max())
                {
                    //std::cout << "nr nodes before = " << bdds[j].nr_nodes() << ", " << bdds[jj].nr_nodes() << ", nodes after = " << intersect.nr_nodes() << " = " <<  (bdds[j] & bdds[jj]).nr_nodes() << "\n";
                    active_bdds[bdd_1] = false;
                    active_bdds[bdd_2] = false;
                    assert(intersect+1 == bdd_collection.nr_bdds());
                    //assert(intersect == active_bdds.size());
                    //active_bdds.push_back(true);
                    bdd_indices[bdd_1] = intersect;
                    bdd_indices[bdd_2] = intersect;
                    current_bdd_intersections.push_back({bdd_1, bdd_2, intersect});
                    ++nr_intersections;
                } 
            }
            
            std::cout << "nr intersections saved = " << nr_intersections << "\n";
            std::vector<size_t> bdds_to_remove;

            // prune out bdds that have are covered by newly constructed ones 
            for(size_t i=0; i<active_bdds.size(); ++i)
                if(active_bdds[i] == false)
                    bdds_to_remove.push_back(i);

            std::vector<size_t> old_bdd_to_new_bdd_nr;
            old_bdd_to_new_bdd_nr.reserve(bdd_collection.nr_bdds());
            size_t c = 0;
            for(size_t i=0; i<active_bdds.size(); ++i)
                if(active_bdds[i] == true)
                    old_bdd_to_new_bdd_nr.push_back(c++);
                else
                    old_bdd_to_new_bdd_nr.push_back(std::numeric_limits<size_t>::max());
            for(size_t i=active_bdds.size(); i<bdd_collection.nr_bdds(); ++i)
                old_bdd_to_new_bdd_nr.push_back(c++);

            //std::unordered_set<std::array<size_t,2>> new_intersected_bdds;
            tsl::robin_set<std::array<size_t,2>> new_intersected_bdds;
            for(const auto [bdd_1, bdd_2] : intersected_bdds)
            {
                assert(bdd_1 < bdd_2);
                const size_t bdd_1_new = old_bdd_to_new_bdd_nr[bdd_1];
                const size_t bdd_2_new = old_bdd_to_new_bdd_nr[bdd_2];
                if(bdd_1_new != std::numeric_limits<size_t>::max() && bdd_2_new != std::numeric_limits<size_t>::max())
                {
                    assert(bdd_1_new < bdd_2_new);
                    assert(new_intersected_bdds.count({bdd_1_new, bdd_2_new}) == 0);
                    new_intersected_bdds.insert({bdd_1_new, bdd_2_new});
                }
            }
            std::swap(intersected_bdds, new_intersected_bdds);

            // prune out intersection bdds that are redundant. Compute edge cover.
            /*
            edge_cover ec;
            for(const bdd_intersection b : current_bdd_intersections)
                ec.add_edge(b.old_bdd[0], b.old_bdd[1], bdd_collection.nr_bdd_nodes(b.new_bdd));
            const auto intersections_to_retain = ec.solve();
            std::cout << "intersections to retain = " << intersections_to_retain.size() << "\n";
            std::cout << "intersections to remove = " << current_bdd_intersections.size() - intersections_to_retain.size() << "\n";
            std::unordered_set<size_t> intersections_to_retain_map;
            for(size_t e : intersections_to_retain)
                intersections_to_retain_map.insert(e);
            for(size_t b = 0; b<current_bdd_intersections.size(); ++b)
                if(intersections_to_retain_map.count(b) == 0)
                    bdds_to_remove.push_back(active_bdds.size() + b);

            bdd_collection.remove(bdds_to_remove.begin(), bdds_to_remove.end());
            */



            std::chrono::steady_clock::time_point coalesce_end = std::chrono::steady_clock::now();
            std::cout << "coalesce time = " << std::chrono::duration_cast<std::chrono::milliseconds>(coalesce_end - coalesce_begin).count() << "[ms]" << std::endl;
            if(bdds_to_remove.size() == 0)
                break;
        }
    }

    /*
    std::tuple<bdd_preprocessor::adjacency_graph, two_dim_variable_array<size_t>, two_dim_variable_array<size_t>>
        bdd_preprocessor::compute_var_adjacency(const size_t bdd_size_limit)
    {
        const auto [bdd_var_adjacency, var_bdd_adjacency] = construct_bdd_var_adjacency(bdd_collection);
        std::unordered_set<std::array<size_t,2>> adjacent_vars;
        for(size_t bdd_nr=0; bdd_nr<bdd_var_adjacency.size(); ++bdd_nr)
        {
            if(bdd_var_adjacency[bdd_nr].size() > bdd_size_limit)
                continue;
            for(size_t i=0; i<bdd_var_adjacency[bdd_nr].size(); ++i)
                for(size_t j=i+1; j<bdd_var_adjacency[bdd_nr].size(); ++j)
                {
                    const size_t v1 = bdd_var_adjacency(bdd_nr,i);
                    const size_t v2 = bdd_var_adjacency(bdd_nr,j);
                    adjacent_vars.insert({std::min(v1,v2), std::max(v1,v2)});
                }
        }

        std::vector<std::array<size_t,2>> edges;
        edges.reserve(adjacent_vars.size());
        for(const auto [i,j] : adjacent_vars)
            edges.push_back({i,j});

        adjacency_graph g(edges.begin(), edges.end());
        return {g, bdd_var_adjacency, var_bdd_adjacency};
    }

    void bdd_preprocessor::coalesce_cliques()
    {
        // TODO: try out tsl::robin_set
        const size_t bdd_size_limit = 0.05 * this->nr_variables;
        const auto [g,bdd_var_adjacency,var_bdd_adjacency] = compute_var_adjacency(bdd_size_limit);
        std::vector<size_t> vars;
        std::vector<size_t> bdds;
        std::unordered_set<size_t> bdd_set;
        size_t nr_max_cliques = 0;
        auto clique_visitor = [&](auto v_begin, auto v_end) {
            nr_max_cliques++;
            return;
            vars.clear();
            for(auto v_it=v_begin; v_it!=v_end; ++v_it)
                vars.push_back(*v_it);
            std::sort(vars.begin(), vars.end());

            bdd_set.clear();
            std::unordered_set<size_t> visited;
            for(const size_t v : vars)
            {
                for(const size_t bdd_nr : var_bdd_adjacency[v])
                {
                    if(bdd_size_limit < bdd_collection.nr_bdd_nodes(bdd_nr))
                        continue;
                    if(visited.count(bdd_nr) > 0)
                        continue;
                    visited.insert(bdd_nr);
                    size_t nr_clique_vars = 0;
                    for(const size_t bdd_var : bdd_var_adjacency[bdd_nr])
                        if(std::find(vars.begin(), vars.end(), bdd_var) != vars.end())
                            nr_clique_vars++;
                    if(nr_clique_vars >= 0.5 * bdd_var_adjacency[bdd_nr].size())
                        bdd_set.insert(bdd_nr);
                }
            }
            if(bdd_set.size() <= 1)
                return;

            // add bdds that have variables in variable set of current bdd set
            std::unordered_set<size_t> covered_variables; 
            for(const size_t bdd_nr : bdd_set)
                for(const size_t v : bdd_var_adjacency[bdd_nr])
                    covered_variables.insert(v);

            for(const size_t v : covered_variables)
            {
                for(const size_t bdd_nr : var_bdd_adjacency[v])
                {
                    if(bdd_set.count(bdd_nr) > 0)
                        continue;
                    const bool variables_subsumed = [&]() {
                    for(const size_t w : bdd_var_adjacency[bdd_nr])
                        if(covered_variables.count(w) == 0)
                            return false;
                    return true;
                    }();
                    if(variables_subsumed)
                        bdd_set.insert(bdd_nr);
                }
            }

            // compute intersection of all bdds
            std::cout << "coalesce clique: ";
            for(size_t bdd_nr : bdd_set)
                std::cout << bdd_nr << ", ";
            std::cout << "\n";
            bdd_collection.bdd_and(bdd_set.begin(), bdd_set.end());
        };
        g.for_each_maximal_clique(clique_visitor);
        std::cout << "nr maximal cliques = " << nr_max_cliques << "\n";
    }
*/

}
