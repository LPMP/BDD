#include <iostream>
#include <unordered_set>
#include "specialized_solvers/graph_matching_input.h"
#include "ILP_input.h"
#include "specialized_solvers/graph_matching_torresani_et_al_grammar.h"
#include "time_measure_util.h"

namespace LPMP {

    std::tuple<ILP_input, std::unordered_map<std::array<size_t,2>, size_t>, std::unordered_map<std::array<size_t,4>, size_t>> construct_graph_matching_ILP(const graph_matching_instance& gm_instance)
    {
        ILP_input ilp;

        // construct simplex factors for linear assignment
        std::vector<std::vector<size_t>> left_assignments;
        std::vector<std::vector<size_t>> right_assignments;
        for(const auto [i,j,c] : gm_instance.linear_assignments)
        {
            if(left_assignments.size() <= i)
                left_assignments.resize(i+1);
            left_assignments[i].push_back(j);
            if(right_assignments.size() <= j)
                right_assignments.resize(j+1);
            right_assignments[j].push_back(i);
        }
        for(auto& l : left_assignments)
        {
            std::sort(l.begin(), l.end());
            // remove duplicates
            l.erase(std::unique(l.begin(), l.end()), l.end());
        }
        for(auto& r : right_assignments)
        {
            std::sort(r.begin(), r.end());
            // remove duplicates
            r.erase(std::unique(r.begin(), r.end()), r.end());
        }
        size_t var_offset = 0;
        std::unordered_map<std::array<size_t,2>,size_t> assignment_map;
        for(size_t l_idx=0; l_idx<left_assignments.size(); ++l_idx)
        {
            const auto& l = left_assignments[l_idx];
            std::vector<int> coeffs(l.size(), 1);
            std::vector<size_t> vars;
            vars.reserve(l.size());
            for(const size_t r : l)
            {
                vars.push_back(var_offset);
                assignment_map.insert({{l_idx,r}, var_offset});
                ++var_offset;
            }
            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
        }

        for(size_t r_idx=0; r_idx<right_assignments.size(); ++r_idx)
        {
            const auto& r = right_assignments[r_idx];
            std::vector<int> coeffs(r.size(), 1);
            std::vector<size_t> vars;
            vars.reserve(r.size());
            for(const size_t l : r)
            {
                assert(assignment_map.count({l,r_idx}) > 0);
                vars.push_back( assignment_map.find({l,r_idx})->second );
            }
            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
        }

        // construct simplex factors for quadratic potentials
        std::unordered_set<std::array<size_t,2>> left_quadratic_simplex;
        std::unordered_set<std::array<size_t,2>> right_quadratic_simplex;
        std::unordered_map<std::array<size_t,4>, size_t> quadratic_variables;
        auto get_quadratic_var = [&](const std::array<size_t,2> i, const std::array<size_t,2> j) {
            assert(i[0] != i[1]);
            assert(j[0] != j[1]);
            assert(quadratic_variables.count({i[0],i[1],j[0],j[1]}) == quadratic_variables.count({i[1],i[0],j[1],j[0]}));
            auto it = quadratic_variables.find({i[0],i[1],j[0],j[1]});
            if(it == quadratic_variables.end())
            {
                quadratic_variables.insert({{i[0],i[1],j[0],j[1]}, var_offset});
                quadratic_variables.insert({{i[1],i[0],j[1],j[0]}, var_offset++});
                return quadratic_variables.find({i[0],i[1],j[0],j[1]})->second;
            }
            return it->second;
        };

        // It turns out that leaving out simplex constraints on pairwise variables makes inference faster
        for(const auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            assert(i[0] != i[1] && j[0] != j[1]);
            if(left_quadratic_simplex.count({i[0],i[1]}) == 0 && left_quadratic_simplex.count({i[1],i[0]}) == 0)
            {
                left_quadratic_simplex.insert({i[0],i[1]});
                std::vector<size_t> vars;
                for(const size_t j0 : left_assignments[i[0]])
                {
                    for(const size_t j1 : left_assignments[i[1]])
                    {
                        if(j0 != j1)
                        {
                            const size_t q_var = get_quadratic_var({i[0], i[1]}, {j0,j1});
                            vars.push_back(q_var);
                        }
                    }
                }
                std::vector<int> coeffs(vars.size(), 1);
                //ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
            }
            if(right_quadratic_simplex.count({j[0],j[1]}) == 0 && right_quadratic_simplex.count({j[1],j[0]}) == 0)
            {
                right_quadratic_simplex.insert({j[0],j[1]});
                std::vector<size_t> vars;
                for(const size_t i0 : right_assignments[j[0]])
                {
                    for(const size_t i1 : right_assignments[j[1]])
                    {
                        if(i0 != i1)
                        {
                            const size_t q_var = get_quadratic_var({i0, i1}, {j[0],j[1]});
                            vars.push_back(q_var);
                        }
                    }
                }
                std::vector<int> coeffs(vars.size(), 1);
                //ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
            }
        }

        // construct marginalization constraints linking linear and quadratic potentials
        for(const auto [i0,i1] : left_quadratic_simplex)
        {
            for(const size_t j0 : left_assignments[i0])
            {
                std::vector<size_t> vars;
                vars.reserve(1 + left_assignments[i1].size());
                vars.push_back( assignment_map.find({i0,j0})->second );
                std::vector<int> coeffs;
                coeffs.reserve(1 + left_assignments[i1].size());
                coeffs.push_back(-1);
                for(const size_t j1 : left_assignments[i1])
                {
                    if(j0 != j1)
                    {
                        // TODO: or check whether i1,i0,j1,j0 is present or put both into quadratic variables?
                        vars.push_back( quadratic_variables.find({i0,i1,j0,j1})->second );
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }

            for(const size_t j1 : left_assignments[i1])
            {
                std::vector<size_t> vars;
                vars.reserve(1 + left_assignments[i0].size());
                vars.push_back( assignment_map.find({i1,j1})->second );
                std::vector<int> coeffs;
                coeffs.reserve(1 + left_assignments[i0].size());
                coeffs.push_back(-1);
                for(const size_t j0 : left_assignments[i0])
                {
                    if(j0 != j1)
                    {
                        // TODO: or check whether i1,i0,j1,j0 is present or put both into quadratic variables?
                        vars.push_back( quadratic_variables.find({i0,i1,j0,j1})->second );
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }
        }

        for(const auto [j0,j1] : right_quadratic_simplex)
        {
            for(const size_t i0 : right_assignments[j0])
            {
                std::vector<size_t> vars;
                vars.reserve(1 + right_assignments[j1].size());
                vars.push_back( assignment_map.find({i0,j0})->second );
                std::vector<int> coeffs;
                coeffs.reserve(1 + right_assignments[j1].size());
                coeffs.push_back(-1);
                for(const size_t i1 : right_assignments[j1])
                {
                    if(i0 != i1 && j0 != j1)
                    {
                        const size_t q_var = get_quadratic_var({i0,i1},{j0,j1});
                        vars.push_back(q_var);
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }

            for(const size_t i1 : right_assignments[j1])
            {
                std::vector<size_t> vars;
                vars.reserve(1 + right_assignments[j0].size());
                vars.push_back( assignment_map.find({i1,j1})->second );
                std::vector<int> coeffs;
                coeffs.reserve(1 + right_assignments[j0].size());
                coeffs.push_back(-1);
                for(const size_t i0 : right_assignments[j0])
                {
                    if(j0 != j1 && i0 != i1)
                    {
                        const size_t q_var = get_quadratic_var({i0,i1},{j0,j1});
                        vars.push_back(q_var);
                        coeffs.push_back(1);
                    }
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }
        }

        std::vector<double> costs;
        for(const auto [i,j,c] : gm_instance.linear_assignments)
        {
            const size_t var = assignment_map.find({i,j})->second;
            // TODO: is very slow
            if(var >= costs.size())
            {
                costs.reserve(2*(var+1));
                costs.resize(var + 1);
            }
            costs[var] = c;
        }

        for(const auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            const size_t var = quadratic_variables.find({i[0],i[1],j[0],j[1]})->second;
            if(var >= costs.size())
            {
                costs.reserve(2*(var+1));
                costs.resize(var + 1);
            }
            costs[var] = c;
        }

        // add objectives
        for(const auto [i,j,c] : gm_instance.linear_assignments)
        {
            const size_t var = assignment_map.find({i,j})->second;
            ilp.add_to_objective(c, var);
        }

        for(const auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            const size_t var = quadratic_variables.find({i[0],i[1],j[0],j[1]})->second;
            ilp.add_to_objective(c, var);
        }

        return {ilp, assignment_map, quadratic_variables};
    }

    ILP_input parse_graph_matching_file(const std::string& filename)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        graph_matching_instance gm_instance;
        std::cout << "[graph matching parser] Parse " << filename << "\n";

        tao::pegtl::file_input input(filename);
        if(!tao::pegtl::parse<TorresaniEtAlInput::grammar, TorresaniEtAlInput::action>(input, gm_instance))
            throw std::runtime_error("[graph matching parser] Could not read file " + filename);

        auto [ilp, linear_vars, quadratic_vars] = construct_graph_matching_ILP(gm_instance);
        return ilp;
    }

    ILP_input parse_graph_matching_string(const std::string& gm_string)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        graph_matching_instance gm_instance;
        std::cout << "[graph matching parser] Parse graph matching string\n";

        tao::pegtl::string_input input(gm_string, "graph matching input");
        if(!tao::pegtl::parse<TorresaniEtAlInput::grammar, TorresaniEtAlInput::action>(input, gm_instance))
            throw std::runtime_error("[graph matching parser] Could not read string");

        auto [ilp, linear_vars, quadratic_vars] = construct_graph_matching_ILP(gm_instance);
        return ilp;
    }

}
