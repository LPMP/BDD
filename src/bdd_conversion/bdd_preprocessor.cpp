#include "bdd_conversion/bdd_preprocessor.h"
#include <iostream>
#include <chrono>
#include <limits>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <cmath>
#include <atomic>
#include "bdd_logging.h"
#include "time_measure_util.h"
#include "two_dimensional_variable_array.hxx"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace LPMP {

    inline size_t getMaximumOccupancy()
    {
#ifdef WITH_CUDA
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        return (deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor)/10;
#else
        return 0;
#endif
    }

    size_t compute_split_length(const BDD::bdd_collection& bdd_collection)
    {
        std::vector<size_t> layer_widths;
        for (size_t bdd_nr = 0; bdd_nr < bdd_collection.nr_bdds(); ++bdd_nr)
        {
            const auto cur_layer_widths = bdd_collection.layer_widths(bdd_nr);
            if (layer_widths.size() < cur_layer_widths.size())
            {
                layer_widths.reserve(2 * cur_layer_widths.size());
                layer_widths.resize(cur_layer_widths.size(), 0);
            }
            for (size_t i = 0; i < cur_layer_widths.size(); ++i)
                layer_widths[i] += cur_layer_widths[i];

        }

        bdd_log << "[bdd preprocessor] Longest BDD has " << layer_widths.size() << " layers\n";
        bdd_log << "[bdd preprocessor] Device can handle " << getMaximumOccupancy() << " concurrent nodes\n";

        // to account for smaller intermediate layer widths take maximum of all subsequent layers into account
        for (ptrdiff_t i = layer_widths.size() - 2; i >= 0; --i)
            layer_widths[i] = std::max(layer_widths[i + 1], layer_widths[i]);

        { // print nr bdd nodes per layer
            int prev = 0;
            int last_num_nodes = layer_widths[0];
            int last_hop = 0;
            for (size_t i = 0; i < layer_widths.size(); i++)
            {
                int current_hop_num_nodes = layer_widths[i];
                if (current_hop_num_nodes != last_num_nodes)
                {
                    bdd_log << "[bdd preprocessor] Layer width: [" << last_hop << " - " << i << "], # BDD nodes: " << last_num_nodes << "\n";
                    last_hop = i;
                    last_num_nodes = current_hop_num_nodes;
                }
                prev = layer_widths[i];
            }
            bdd_log << "[bdd preprocessor] Layer width: [" << last_hop << " - " << layer_widths.size() - 1 << "], # BDD nodes: " << last_num_nodes << "\n";
        }

        const size_t maximum_occupancy = getMaximumOccupancy();

        auto split_layer_widths = [&](const size_t split_length)
        {
            std::vector<size_t> new_layer_widths(split_length, 0);

            for (size_t i = 0; i < split_length; ++i)
                for (size_t j = i; j < layer_widths.size(); j += split_length)
                    new_layer_widths[i] += layer_widths[j];

            return new_layer_widths;
        };

        auto avg_occupancy = [](const size_t parallelism, const std::vector<size_t>& layer_widths) {
            double avg_occ = 0.0;
            for(size_t i=0; i<layer_widths.size(); ++i)
                avg_occ += double(std::min(layer_widths[i], parallelism)) / double(parallelism);
            return avg_occ / layer_widths.size();
        };

        // find largest split length such that occupancy >= 0.5
        double occupancy = avg_occupancy(getMaximumOccupancy(), layer_widths);
        int split_length = layer_widths.size();
        for(; split_length >= 200; --split_length)
        {
            const auto new_layer_widths = split_layer_widths(split_length);
            if(avg_occupancy(getMaximumOccupancy(), new_layer_widths) >= 0.5)
                break;
        }

        /*
        size_t max_length_bdd = [&]() -> size_t
        {
            if (*std::max_element(layer_widths.begin(), layer_widths.end()) < 2048 || bdd_collection.nr_bdds() < 128)
            {
                bdd_log << "[bdd preprocessor] Too {few|small} BDDs, do not split\n";
                return std::numeric_limits<size_t>::max();
            }
            for (size_t i = 200; i < std::max(std::ptrdiff_t(layer_widths.size()) - 50, std::ptrdiff_t(0)); ++i)
                if (layer_widths[i] < maximum_occupancy)
                    return i;
            return std::numeric_limits<size_t>::max();
        }();
        */

        bdd_log << "[bdd preprocessor] Split BDDs longer than " << split_length << " layers\n";

        return split_length;
    }

    two_dim_variable_array<size_t> bdd_preprocessor::add_ilp(const ILP_input& input, const bool normalize, const bool split_long_bdds, const bool add_split_implication_bdd, const size_t split_length)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(bdd_collection.nr_bdds() == 0);
        // first transform linear inequalities into BDDs
        bdd_log << "[bdd preprocessor] convert " << input.constraints().size() << " linear inequalities.\n";
        if(normalize)
            bdd_log << "[bdd preprocessor] normalize constraints\n";

        std::vector<size_t> ineq_nrs;
        two_dim_variable_array<size_t> bdd_nrs;

#ifdef _OPENMP
        const size_t nr_threads = omp_get_max_threads();
#else
        const size_t nr_threads = 1;
#endif
        bdd_log << "[bdd preprocessor] #threads = " << nr_threads << "\n";

        // for variable copies when using coefficient decomposition transformation to BDDs
        std::atomic<size_t> extra_var_counter = input.nr_variables();

        // TODO: tid based construction not needed anymore, do directly through openmp for loop sharing
#pragma omp parallel for ordered schedule(static) num_threads(nr_threads)
        for(size_t tid=0; tid<nr_threads; ++tid)
        {
            std::vector<int> coefficients;
            std::vector<std::size_t> variables;
            std::vector<size_t> cur_ineq_nrs;
            two_dim_variable_array<size_t> cur_bdd_nrs;
            std::vector<size_t> cur_bdds_to_remove;
            BDD::bdd_mgr bdd_mgr;
            bdd_converter converter(bdd_mgr);
            BDD::bdd_collection cur_bdd_collection;

            auto make_qbdd = [&](const size_t bdd_nr) {
                assert(bdd_nr + 1 == cur_bdd_collection.nr_bdds());
                if(!cur_bdd_collection.is_qbdd(bdd_nr))
                {
                    const size_t new_bdd_nr = cur_bdd_collection.make_qbdd(bdd_nr);
                    cur_bdd_collection.remove(bdd_nr);
                    assert(cur_bdd_collection.is_qbdd(bdd_nr));
                }
            };

            const size_t first_constr = input.constraints().size()/nr_threads * tid;
            const size_t last_constr = (tid+1 == nr_threads) ? input.constraints().size() : (input.constraints().size()/nr_threads) * (tid+1);
            if(tid + 1 == nr_threads)
                assert(last_constr == input.constraints().size());

            for(size_t c=first_constr; c<last_constr; ++c)
            {
                const auto constraint = [&]() {
                    auto constraint = input.constraints()[c];
                    if(normalize && !constraint.is_normalized())
                        constraint.normalize();
                    return constraint;
                }();
                variables.clear();
                if(constraint.is_simplex())
                {
                    const size_t bdd_nr = cur_bdd_collection.simplex_constraint(constraint.coefficients.size());
                    for(size_t monomial_idx=0; monomial_idx<constraint.monomials.size(); ++monomial_idx)
                    {
                        const size_t var = constraint.monomials(monomial_idx, 0);
                        variables.push_back(var);
                    }
                    cur_bdd_collection.rebase(bdd_nr, variables.begin(), variables.end());
                    assert(cur_bdd_collection.is_qbdd(bdd_nr));
                    assert(cur_bdd_collection.variables(bdd_nr) == variables);

                    cur_ineq_nrs.push_back(c);
                    std::array<size_t,1> bdd_nr_array = {bdd_nr};
                    cur_bdd_nrs.push_back(bdd_nr_array.begin(), bdd_nr_array.end());
                }
                else if(constraint.is_linear())
                {
                    assert(constraint.monomials.size() == constraint.coefficients.size());
                    for(size_t monomial_idx=0; monomial_idx<constraint.monomials.size(); ++monomial_idx)
                    {
                        const size_t var = constraint.monomials(monomial_idx, 0);
                        const int coeff = constraint.coefficients[monomial_idx];
                        variables.push_back(var);
                    }

                    const size_t nr_vars = constraint.coefficients.size();
                    assert(constraint.coefficients.size() > 0);
                    const int max_coeff = std::max(
                            *std::max_element(constraint.coefficients.begin(), constraint.coefficients.end()),
                            - *std::min_element(constraint.coefficients.begin(), constraint.coefficients.end())
                            );
                    if(nr_vars <= 64 || max_coeff <= 100) // convert to BDD directly
                    {
                        BDD::node_ref bdd = converter.convert_to_bdd(constraint.coefficients, constraint.ineq, constraint.right_hand_side);
                        if(bdd.is_topsink())
                            continue;
                        else if(bdd.is_botsink())
                            throw std::runtime_error("problem is infeasible");
                        const size_t bdd_nr = cur_bdd_collection.add_bdd(bdd);
                        cur_bdd_collection.reorder(bdd_nr);
                        make_qbdd(bdd_nr);
                        assert(cur_bdd_collection.is_reordered(bdd_nr));
                        cur_bdd_collection.rebase(bdd_nr, variables.begin(), variables.end());
                        assert(cur_bdd_collection.is_qbdd(bdd_nr));
                        assert(cur_bdd_collection.variables(bdd_nr) == variables);

                        cur_ineq_nrs.push_back(c);
                        std::array<size_t,1> bdd_nr_array = {bdd_nr};
                        cur_bdd_nrs.push_back(bdd_nr_array.begin(), bdd_nr_array.end());
                    }
                    else // use coefficient decomposition
                    {
                        if(normalize)
                            throw std::runtime_error("coefficient decomposition BDDs may not be sorted w.r.t. variable indices"); 

                        bdd_log << "[bdd preprocessor] convert inequality " << constraint.identifier << " through coefficient decomposition. max coeff: "<< max_coeff << ", nr_vars: " << nr_vars << "\n";
                        input.write_lp(bdd_log, constraint);
                        auto [bdd, var_split] = converter.coefficient_decomposition_convert_to_bdd(constraint.coefficients, constraint.ineq, constraint.right_hand_side);

                        if(bdd.is_topsink())
                        {
                            assert(false); // bdd nrs must be recorded
                            continue;
                        }
                        else if(bdd.is_botsink())
                            throw std::runtime_error("problem is infeasible");

                        const size_t bdd_nr = cur_bdd_collection.add_bdd(bdd);
                        cur_bdd_collection.reorder(bdd_nr);
                        make_qbdd(bdd_nr);
                        assert(cur_bdd_collection.is_reordered(bdd_nr));

                        std::vector<size_t> copy_variables(var_split.data().size(), std::numeric_limits<size_t>::max());

                        assert(variables.size() == var_split.size());
                        for(size_t i=0; i<variables.size(); ++i)
                        {
                            std::vector<size_t> var_copy_equal_vars;
                            assert(var_split.size(i) > 0);
                            if(var_split.size(i) == 1)
                            {
                                assert(var_split(i,0) < copy_variables.size());
                                copy_variables[var_split(i,0)] = variables[i];
                            }
                            else
                            {
                                // additionally add BDDs for equality between decomposed variables
                                var_copy_equal_vars.push_back(variables[i]);
                                for(size_t j=0; j<var_split.size(i); ++j)
                                {
                                    const size_t new_var = extra_var_counter++;
                                    assert(var_split(i,j) < copy_variables.size());
                                    assert(copy_variables[var_split(i,j)] == std::numeric_limits<size_t>::max());
                                    copy_variables[var_split(i,j)] = new_var;
                                    var_copy_equal_vars.push_back(new_var);
                                }
                                std::sort(var_copy_equal_vars.begin(), var_copy_equal_vars.end());
                                assert(std::unique(var_copy_equal_vars.begin(), var_copy_equal_vars.end()) == var_copy_equal_vars.end());
                                const size_t equal_bdd_nr = cur_bdd_collection.all_equal_constraint(var_copy_equal_vars.size());
                                cur_bdd_collection.rebase(equal_bdd_nr, var_copy_equal_vars.begin(), var_copy_equal_vars.end());
                                assert(cur_bdd_collection.is_reordered(equal_bdd_nr));
                                assert(cur_bdd_collection.is_qbdd(equal_bdd_nr));
                            }
                        }

                        for(size_t i=0; i<copy_variables.size(); ++i)
                            assert(copy_variables[i] != std::numeric_limits<size_t>::max());

                        cur_bdd_collection.rebase(bdd_nr, copy_variables.begin(), copy_variables.end());
                        assert(cur_bdd_collection.is_qbdd(bdd_nr));

                        cur_ineq_nrs.push_back(c);
                        std::vector<size_t> new_bdd_nrs;
                        for(size_t i=0; i<cur_bdd_collection.nr_bdds(); ++i)
                            new_bdd_nrs.push_back(bdd_nr + i);
                        cur_bdd_nrs.push_back(new_bdd_nrs.begin(), new_bdd_nrs.end());
                    }
                }
                else if(constraint.distinct_variables()) // nonlinear BDD
                {
                    if(normalize)
                        throw std::runtime_error("nonlinear BDDs may not be sorted w.r.t. variable indices"); 
                    std::vector<size_t> monomial_degrees;
                    monomial_degrees.reserve(constraint.coefficients.size());
                    for(size_t monomial_idx=0; monomial_idx<constraint.monomials.size(); ++monomial_idx)
                        monomial_degrees.push_back(constraint.monomials.size(monomial_idx));
                    BDD::node_ref bdd = converter.convert_nonlinear_to_bdd(monomial_degrees, constraint.coefficients, constraint.ineq, constraint.right_hand_side);

                    assert(!bdd.is_terminal());
                    const size_t bdd_nr = cur_bdd_collection.add_bdd(bdd);
                    cur_bdd_collection.reorder(bdd_nr);
                    assert(cur_bdd_collection.is_reordered(bdd_nr));

                    for(size_t monomial_idx=0; monomial_idx<constraint.monomials.size(); ++monomial_idx)
                    {
                        for(size_t i=0; i<constraint.monomials.size(monomial_idx); ++i)
                        {
                            const size_t var = constraint.monomials(monomial_idx, i);
                            variables.push_back(var);
                        }
                    }

                    make_qbdd(bdd_nr);
                    cur_bdd_collection.rebase(bdd_nr, variables.begin(), variables.end());

                    cur_ineq_nrs.push_back(c);
                    std::array<size_t,2> bdd_nr_array = {bdd_nr};
                    cur_bdd_nrs.push_back(bdd_nr_array.begin(), bdd_nr_array.end());
                }
                else
                {
                    throw std::runtime_error("only linear constraints supported");
                }
            }
            // add everything to one bdd collection, store mapping from inequalities to bdd numbers
#pragma omp ordered 
            {
                const size_t bdd_nr_offset = bdd_collection.nr_bdds();
                bdd_collection.append(cur_bdd_collection);
                assert(bdd_collection.nr_bdds() == cur_bdd_collection.nr_bdds() + bdd_nr_offset);

                assert(cur_ineq_nrs.size() == cur_bdd_nrs.size());

                // record corresponding inequality of each BDD
                for(size_t c=0; c<cur_bdd_nrs.size(); ++c)
                {
                    for(size_t j=0; j<cur_bdd_nrs.size(c); ++j)
                    {
                        cur_bdd_nrs(c,j) += bdd_nr_offset;
                        assert(cur_bdd_nrs(c,j) >= bdd_nr_offset);
                    }
                    bdd_nrs.push_back(cur_bdd_nrs.begin(c), cur_bdd_nrs.end(c));
                }
                ineq_nrs.insert(ineq_nrs.end(), cur_ineq_nrs.begin(), cur_ineq_nrs.end());
                assert(ineq_nrs.size() == bdd_nrs.size());
            }
        }

        // reorder bdd nrs to make them consecutive w.r.t. inequality numbers
        two_dim_variable_array<size_t> ineq_to_bdd_nrs;
        std::vector<size_t> inv_ineq_nrs(ineq_nrs.size());
        for(size_t c=0; c<ineq_nrs.size(); ++c)
            inv_ineq_nrs[ineq_nrs[c]] = c;

        for(size_t c=0; c<ineq_nrs.size(); ++c)
            ineq_to_bdd_nrs.push_back(bdd_nrs.begin(inv_ineq_nrs[c]), bdd_nrs.end(inv_ineq_nrs[c]));

        assert(ineq_to_bdd_nrs.size() == input.constraints().size());

        // split long bdds
        // TODO: ineq_to_bdd_nrs will not be valid anymore! This cannot work with the learned solver.
        if(split_long_bdds || split_length < std::numeric_limits<size_t>::max())
        {
            const size_t max_length_bdd = [&]()
            {
                if (split_length < std::numeric_limits<size_t>::max())
                {
                    bdd_log << "[bdd preprocessor] force split BDDs longer than " << split_length << "\n";
                    return split_length;
                }
                else
                    return compute_split_length(bdd_collection);
            }();

            if (max_length_bdd < std::numeric_limits<size_t>::max())
            {
                std::vector<size_t> bdds_to_remove;
                const size_t nr_orig_bdds = bdd_collection.nr_bdds();
                for (size_t bdd_nr = 0; bdd_nr < nr_orig_bdds; ++bdd_nr)
                {
                    if (bdd_collection.nr_variables(bdd_nr) > max_length_bdd)
                    {
                        const auto [new_bdd_nrs, new_aux_var] = bdd_collection.split_qbdd(bdd_nr, max_length_bdd, extra_var_counter, add_split_implication_bdd);
                        for(const size_t new_bdd_nr : new_bdd_nrs)
                            assert(bdd_collection.is_qbdd(new_bdd_nr));
                        assert(new_bdd_nrs.size() > 0);
                        extra_var_counter = new_aux_var;
                        if(new_bdd_nrs.size() > 1)
                        {
                            bdds_to_remove.push_back(bdd_nr);
                            assert(bdd_collection.nr_bdds() == new_bdd_nrs.back() + 1);
                        }
                        else
                            assert(new_bdd_nrs.size() == 1 && new_bdd_nrs[0] == bdd_nr);
                    }
                }
                for (size_t bdd_nr = 0; bdd_nr < bdd_collection.nr_bdds(); ++bdd_nr)
                    assert(bdd_collection.is_qbdd(bdd_nr));

                bdd_log << "[bdd preprocessor] Split " << bdds_to_remove.size() << " BDDs\n";
                bdd_collection.remove(bdds_to_remove.begin(), bdds_to_remove.end());
            }
        }

        for (size_t bdd_nr = 0; bdd_nr < bdd_collection.nr_bdds(); ++bdd_nr)
            assert(bdd_collection.is_qbdd(bdd_nr));

        bdd_log << "[bdd preprocessor] final #BDDs = " << bdd_collection.nr_bdds() << "\n";

        return ineq_to_bdd_nrs;
    }

}
