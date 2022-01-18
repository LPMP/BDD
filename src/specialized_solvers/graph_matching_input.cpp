#include <iostream>
#include <unordered_set>
#include "specialized_solvers/graph_matching_input.h"
#include "ILP_input.h"
#include "specialized_solvers/graph_matching_torresani_et_al_grammar.h"
#include "time_measure_util.h"
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace LPMP {

    std::tuple<ILP_input, std::unordered_map<std::array<size_t,2>, size_t>, std::unordered_map<std::array<size_t,4>, size_t>> construct_graph_matching_ILP(const graph_matching_instance& gm_instance)
    {
        ILP_input ilp;

        constexpr size_t no_assignment = std::numeric_limits<size_t>::max();

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

        std::cout << "[graph matching parser] Construct " << left_assignments.size() << " simplex factors ->\n";

        for(auto& l : left_assignments)
        {
            l.push_back(no_assignment);
            std::sort(l.begin(), l.end());
            // remove duplicates
            l.erase(std::unique(l.begin(), l.end()), l.end());
        }

        std::cout << "[graph matching parser] Construct " << right_assignments.size() << " simplex factors <-\n";

        for(auto& r : right_assignments)
        {
            r.push_back(no_assignment);
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
            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 1);
        }

        for(size_t r_idx=0; r_idx<right_assignments.size(); ++r_idx)
        {
            const auto& r = right_assignments[r_idx];
            std::vector<int> coeffs(r.size(), 1);
            std::vector<size_t> vars;
            vars.reserve(r.size());
            for(const size_t l : r)
            {
                if(l == no_assignment)
                {
                    assert(assignment_map.count({l,r_idx}) == 0);
                    assignment_map.insert({{l,r_idx}, var_offset++});
                }
                else
                {
                    assert(assignment_map.count({l,r_idx}) > 0);
                }
                vars.push_back( assignment_map.find({l,r_idx})->second );
            }
            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 1);
        }

        // construct simplex factors for quadratic potentials
        // count how many quadratic assignments are in each quadratic potential
        struct left_pairwise_var { size_t q_ILP_var; size_t j0; size_t j1; };
        std::unordered_map<std::array<size_t,2>, std::vector<left_pairwise_var>> left_quadratic_simplex;
        struct right_pairwise_var { size_t q_ILP_var; size_t i0; size_t i1; };
        std::unordered_map<std::array<size_t,2>, std::vector<right_pairwise_var>> right_quadratic_simplex;
        std::unordered_map<std::array<size_t,4>, size_t> quadratic_variables;

        auto get_quadratic_var = [&](std::array<size_t,2> i, std::array<size_t,2> j) {
            assert(i[0] != i[1] || i[0] == no_assignment);
            assert(j[0] != j[1] || j[0] == no_assignment);
            assert(std::max(i[0], i[1]) < left_assignments.size() || std::max(i[0], i[1]) == no_assignment);
            assert(std::max(j[0], j[1]) < right_assignments.size() || std::max(j[0], j[1]) == no_assignment);
            if(i[0] > i[1])
            {
                std::swap(i[0], i[1]);
                std::swap(j[0], j[1]);
            }

            assert(i[0] < i[1] || i[0] == no_assignment);
            assert(i[1] < left_assignments.size() || i[1] == no_assignment);

            auto it = quadratic_variables.find({i[0],i[1],j[0],j[1]});
            if(it != quadratic_variables.end())
            {
                return it->second;
            }
            else
            {
                quadratic_variables.insert({{i[0],i[1],j[0],j[1]}, var_offset++});
                return quadratic_variables.find({i[0],i[1],j[0],j[1]})->second;
            }
        };

        // It turns out that leaving out simplex constraints on pairwise variables makes inference faster
        for(const auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            // sometimes problems are specified with inadmissible quadratic assignments, leave them out
            assert(std::max({i[0],i[1],j[0],j[1]}) < no_assignment);
            if(i[0] == i[1] || j[0] == j[1])
                continue;

            assert(i[0] != i[1] && j[0] != j[1] && std::max({i[0], i[1], j[0], j[1]}) != no_assignment);

            const size_t i0 = std::min(i[0], i[1]);
            const size_t i1 = std::max(i[0], i[1]);
            if(left_quadratic_simplex.count({i0,i1}) == 0)
            {
                left_quadratic_simplex.insert({{i0,i1}, {}});
                //std::vector<size_t> vars;
                for(const size_t j0 : left_assignments[i0])
                {
                    for(const size_t j1 : left_assignments[i1])
                    {
                        if(j0 != j1)
                        {
                            //const size_t q_var = get_quadratic_var({i0, i1}, {j0, j1});
                            //vars.push_back(q_var);
                        }
                    }
                }
                //std::vector<int> coeffs(vars.size(), 1);
                //ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
            }

            // i and j must be consistent
            if(i[0] < i[1])
                left_quadratic_simplex.find({i0,i1})->second.push_back({get_quadratic_var(i,j), j[0], j[1]});
            else
                left_quadratic_simplex.find({i0,i1})->second.push_back({get_quadratic_var(i,j), j[1], j[0]});

            const size_t j0 = std::min(j[0], j[1]);
            const size_t j1 = std::max(j[0], j[1]);
            if(right_quadratic_simplex.count({j0,j1}) == 0)
            {
                right_quadratic_simplex.insert({{j0,j1}, {}});

                //std::vector<size_t> vars;
                for(const size_t i0 : right_assignments[j0])
                {
                    for(const size_t i1 : right_assignments[j1])
                    {
                        if(i0 != i1)
                        {
                            //const size_t q_var = get_quadratic_var({i0, i1}, {j0, j1});
                            //vars.push_back(q_var);
                        }
                    }
                }
                //std::vector<int> coeffs(vars.size(), 1);
                //ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);
            }

            // i and j must be consistent
            if(j[0] < j[1])
            {
                right_quadratic_simplex.find({j0,j1})->second.push_back({get_quadratic_var(i,j), i[0], i[1]});
            }
            else
            {
                right_quadratic_simplex.find({j0,j1})->second.push_back({get_quadratic_var(i,j), i[1], i[0]});
            }
        }

        std::vector<size_t> vars;
        std::vector<int> coeffs;

        const auto [construct_left_quadratic, construct_right_quadratic] = [&]() -> std::tuple<bool, bool> {
            if(10 * left_quadratic_simplex.size() <= right_quadratic_simplex.size())
                return {true, false};
            if(10 * right_quadratic_simplex.size() <= left_quadratic_simplex.size())
                return {true, false};
            return {true, true};
        }();

        // construct marginalization constraints linking linear and quadratic potentials
        if(construct_left_quadratic)
        {
            std::cout << "[graph matching solver] Construct " << left_quadratic_simplex.size() << " quadratic simplex factors ->\n";
            for(const auto& [i,quadratic_vars] : left_quadratic_simplex)
            {
                const size_t i0 = i[0];
                const size_t i1 = i[1];
                assert(i0 < i1);
                //if(quadratic_vars.size() >= 0.1 * left_assignments[i0].size() * left_assignments[i0].size()) // full quadratic potential
                //{
                for(const size_t j0 : left_assignments[i0])
                {
                    vars.clear();
                    coeffs.clear();
                    vars.push_back( assignment_map.find({i0,j0})->second );
                    coeffs.push_back(-1);
                    for(const size_t j1 : left_assignments[i1])
                    {
                        if(j0 != j1 || j0 == no_assignment)
                        {
                            vars.push_back(get_quadratic_var({i0,i1},{j0,j1}));
                            coeffs.push_back(1);
                        }
                    }
                    ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
                }

                for(const size_t j1 : left_assignments[i1])
                {
                    vars.clear();
                    coeffs.clear();
                    vars.push_back( assignment_map.find({i1,j1})->second );
                    coeffs.push_back(-1);
                    for(const size_t j0 : left_assignments[i0])
                    {
                        if(j0 != j1 || j0 == no_assignment)
                        {
                            vars.push_back(get_quadratic_var({i0,i1},{j0,j1}));
                            coeffs.push_back(1);
                        }
                    }
                    ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
                }
                //}
                //else // sparse quadratic potential
                //{
                //    throw std::runtime_error("sparse left quadratic potential not supported");
                //}
            }
        }

        if(construct_right_quadratic)
        {
            std::cout << "[graph matching solver] Construct " << right_quadratic_simplex.size() << " quadratic simplex factors <-\n";
            for(const auto& [j,quadratic_vars] : right_quadratic_simplex)
            {
                const size_t j0 = j[0];
                const size_t j1 = j[1];
                assert(j0 < j1);
                assert(right_assignments[j0].size() > 0);
                assert(right_assignments[j1].size() > 0);

                //if(quadratic_vars.size() >= 0.1 * right_assignments[j0].size() * right_assignments[j0].size()) // full quadratic potential
                //{
                for(const size_t i0 : right_assignments[j0])
                {
                    vars.clear();
                    coeffs.clear();
                    vars.push_back( assignment_map.find({i0,j0})->second );
                    coeffs.push_back(-1);
                    for(const size_t i1 : right_assignments[j1])
                    {
                        if(i0 != i1 || i0 == no_assignment)
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
                    vars.clear();
                    coeffs.clear();
                    vars.push_back( assignment_map.find({i1,j1})->second );
                    coeffs.push_back(-1);
                    for(const size_t i0 : right_assignments[j0])
                    {
                        if(i0 != i1 || j0 == no_assignment)
                        {
                            const size_t q_var = get_quadratic_var({i0,i1},{j0,j1});
                            vars.push_back(q_var);
                            coeffs.push_back(1);
                        }
                    }
                    ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
                }
                //}
                /*
                   else // do sparse pairwise potential
                   {
                   std::vector<size_t> constraint_group;
                   vars.clear();
                   coeffs.clear();

                   for(const auto [q_ILP_var, i0, i1] : quadratic_vars)
                   {
                   coeffs.push_back(1);
                   vars.push_back(q_ILP_var);
                   }
                   ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 1);

                   for(const auto [q_ILP_var, i0, i1] : quadratic_vars)
                   {
                   constraint_group.clear();
                // x_i0,i1,j0,j1 >= x_i0,j0 + x_i1,j1 - 1
                assert(q_ILP_var == get_quadratic_var({i0,i1}, {j0,j1}));
                coeffs.clear();
                vars.clear();
                vars.push_back(q_ILP_var);
                coeffs.push_back(1);

                assert(assignment_map.count({i0,j0}) > 0);
                vars.push_back(assignment_map.find({i0,j0})->second);
                coeffs.push_back(-1);

                assert(assignment_map.count({i1,j1}) > 0);
                vars.push_back(assignment_map.find({i1,j1})->second);
                coeffs.push_back(-1);

                constraint_group.push_back( ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::greater_equal, -1) );
                assert(constraint_group.back() < ilp.constraints().size());

                // x_i0,i1,j0,j1 <= x_i0,j0
                coeffs.clear();
                vars.clear();
                coeffs.push_back(1);
                vars.push_back(q_ILP_var);
                coeffs.push_back(-1);
                vars.push_back(assignment_map.find({i0,j0})->second);
                constraint_group.push_back( ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 0) );
                assert(constraint_group.back() < ilp.constraints().size());

                // x_i0,i1,j0,j1 <= x_i1,j1
                coeffs.clear();
                vars.clear();
                coeffs.push_back(1);
                vars.push_back(q_ILP_var);
                coeffs.push_back(-1);
                vars.push_back(assignment_map.find({i1,j1})->second);
                constraint_group.push_back( ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::smaller_equal, 0) );
                assert(constraint_group.back() < ilp.constraints().size());

                ilp.add_constraint_group(constraint_group.begin(), constraint_group.end());
                }

                }
                */
            }
        }

        size_t max_var = 0;
        for(const auto [i,j,c] : gm_instance.linear_assignments)
        {
            const size_t var = assignment_map.find({i,j})->second;
            max_var = std::max(var, max_var);
        }
        for(const auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            const size_t var = get_quadratic_var(i,j);
            max_var = std::max(var, max_var);
        }

        std::vector<double> costs(max_var+1, 0);

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
            const size_t var = get_quadratic_var({i[0],i[1]},{j[0],j[1]});
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

        for(auto [i,j,c] : gm_instance.quadratic_assignments)
        {
            if(i[0] > i[1])
            {
                std::swap(i[0], i[1]);
                std::swap(j[0], j[1]);
            }
            assert(i[0] < i[1]);
            const size_t var = get_quadratic_var({i[0],i[1]},{j[0],j[1]});
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

        std::cout << "[graph matching parser] problem has " << gm_instance.linear_assignments.size() << " linear and " << gm_instance.quadratic_assignments.size() << " quadratic assignments\n";

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
