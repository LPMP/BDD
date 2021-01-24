#include "ILP_input.h"
#include <Eigen/Eigen>
#include "cuthill-mckee.h"
#include "bfs_ordering.hxx"
#include "minimum_degree_ordering.hxx"
#include "time_measure_util.h"
#include <iostream>

namespace LPMP { 

    bool ILP_input::var_exists(const std::string& var) const
    {
        return var_name_to_index_.count(var) > 0;
    }

    size_t ILP_input::get_var_index(const std::string& var) const
    {
        assert(var_exists(var));
        return var_name_to_index_.find(var)->second;
    }

    std::string ILP_input::get_var_name(const size_t index) const
    {
        assert(index < var_index_to_name_.size());
        return var_index_to_name_[index];
    }

    size_t ILP_input::add_new_variable(const std::string& var)
    {
        assert(!var_exists(var));
        const size_t var_index = var_name_to_index_.size();
        var_name_to_index_.insert({var, var_index});
        assert(var_index_to_name_.size() == var_index);
        var_index_to_name_.push_back(var);
        if(objective_.size() <= var_index) // variables with 0 objective coefficient need not appear in objective line!
            objective_.resize(var_index+1,0.0);
        return var_index;
    }

    size_t ILP_input::get_or_create_variable_index(const std::string& var)
    {
        if(var_exists(var))
            return get_var_index(var);
        else 
            return add_new_variable(var); 
    }

    size_t ILP_input::nr_variables() const
    {
        return var_name_to_index_.size();
    }

    void ILP_input::add_to_objective(const double coefficient, const std::string& var)
    {
        add_to_objective(coefficient, get_or_create_variable_index(var));
    }

    void ILP_input::add_to_objective(const double coefficient, const size_t var)
    {
        assert(var < objective_.size());
        objective_[var] += coefficient;
    }

    double ILP_input::objective(const size_t var) const
    {
        if(var >= nr_variables())
            throw std::runtime_error("variable not present");
        if(var >= objective_.size())
            return 0.0;
        return objective_[var];
    }

    double ILP_input::objective(const std::string& var) const
    {
        return objective(get_var_index(var));
    }

    void ILP_input::begin_new_inequality()
    {
        linear_constraints_.push_back({});
    }

    void ILP_input::set_inequality_type(const inequality_type ineq)
    {
        assert(linear_constraints_.size() > 0);
        linear_constraints_.back().ineq = ineq;
    }

    void ILP_input::add_to_constraint(const int coefficient, const size_t var)
    {
        assert(linear_constraints_.size() > 0);
        auto& constr = linear_constraints_.back();
        constr.variables.push_back({coefficient, var}); 

        if(constr.variables.size() > 1)
            if(constr.variables.back() < constr.variables[constr.variables.size()-2])
                constr.normalize();
    }
    void ILP_input::add_to_constraint(const int coefficient, const std::string& var)
    {
        add_to_constraint(coefficient, get_or_create_variable_index(var));
    }

    void ILP_input::set_right_hand_side(const int x)
    {
        assert(linear_constraints_.size() > 0);
        linear_constraints_.back().right_hand_side = x;
    } 

    size_t ILP_input::nr_constraints() const
    {
        return linear_constraints_.size();
    }

    bool ILP_input::preprocess()
    {
        for (auto it = linear_constraints_.begin(); it != linear_constraints_.end(); it++)
        {
            bool remove = false;

            // empty constraints
            if (it->variables.empty())
            {
                // feasibility check
                switch (it->ineq)
                {
                    case inequality_type::smaller_equal:
                        if(it->right_hand_side < 0)
                            return false;
                        break;
                    case inequality_type::greater_equal:
                        if(it->right_hand_side > 0)
                            return false;
                        break;
                    case inequality_type::equal:
                        if(it->right_hand_side != 0)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
                remove = true;
            }

            // variable bounds and fixations
            if (it->variables.size() == 1)
            {
                auto variable = it->variables.back();
                switch (it->ineq)
                {
                    case inequality_type::smaller_equal:
                        if(std::min(variable.coefficient, 0) > it->right_hand_side)
                            return false;
                        if(std::max(variable.coefficient, 0) <= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::greater_equal:
                        if(std::max(variable.coefficient, 0) < it->right_hand_side)
                            return false;
                        if(std::min(variable.coefficient, 0) >= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::equal:
                        if(it->right_hand_side != 0 && it->right_hand_side != variable.coefficient)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
            }

            // TODO implement preprocessing of variable/group fixations (e.g. x = 1, x + y = 0)

            // remove redundant constraint
            if (remove)
            {
                *it = linear_constraints_.back();
                linear_constraints_.pop_back();
                it--;    
            }
        }

        return true;
    }

    inline two_dim_variable_array<size_t> ILP_input::variable_adjacency_matrix() const
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<Eigen::Triplet<int>> var_constraint_adjacency_list;

        for(size_t i=0; i<this->linear_constraints_.size(); ++i) {
            const auto& l = this->linear_constraints_[i];
            for(const auto& v : l.variables) {
                var_constraint_adjacency_list.push_back({v.var, i, 1});
            }
        }

        Eigen::SparseMatrix<int> A(this->nr_variables(), this->linear_constraints_.size());
        A.setFromTriplets(var_constraint_adjacency_list.begin(), var_constraint_adjacency_list.end());
        const Eigen::SparseMatrix<int> adj_matrix = A*A.transpose();
        assert(adj_matrix.cols() == nr_variables() && adj_matrix.rows() == nr_variables());

        std::vector<size_t> adjacency_size(nr_variables(), 0);

        for(size_t i=0; i<adj_matrix.outerSize(); ++i) {
            for(typename Eigen::SparseMatrix<int>::InnerIterator it(adj_matrix,i); it; ++it) {
                if(it.value() > 0 && i != it.index()) {
                    adjacency_size[i]++;
                }
            }
        }

        two_dim_variable_array<size_t> adjacency(adjacency_size.begin(), adjacency_size.end());
        std::fill(adjacency_size.begin(), adjacency_size.end(), 0);

        for(size_t i=0; i<adj_matrix.outerSize(); ++i) {
            for(typename Eigen::SparseMatrix<int>::InnerIterator it(adj_matrix,i); it; ++it) {
                if(it.value() > 0 && i != it.index()) {
                    adjacency(i,adjacency_size[i]++) = it.index();
                }
            }
        }

        for(size_t i=0; i<adjacency.size(); ++i) {
            assert(std::is_sorted(adjacency[i].begin(), adjacency[i].end()));
        }

        return adjacency;
    }

    /*
       inline two_dim_variable_array<size_t> ILP_input::variable_adjacency_matrix() const
       {
    //std::unordered_set<std::array<size_t,2>> adjacent_vars;
    tsl::robin_set<std::array<size_t,2>> adjacent_vars;
    for(const auto& l : this->linear_constraints_) {
    for(size_t i=0; i<l.variables.size(); ++i) {
    for(size_t j=i+1; j<l.variables.size(); ++j) {
    const size_t var1 = l.variables[i].var;
    const size_t var2 = l.variables[j].var;
    adjacent_vars.insert({std::min(var1,var2), std::max(var1, var2)});
    }
    }
    }

    std::vector<size_t> adjacency_size(this->nr_variables(),0);
    for(const auto [i,j] : adjacent_vars) {
    ++adjacency_size[i];
    ++adjacency_size[j];
    }

    two_dim_variable_array<size_t> adjacency(adjacency_size.begin(), adjacency_size.end());
    std::fill(adjacency_size.begin(), adjacency_size.end(), 0);
    for(const auto e : adjacent_vars) {
    const auto [i,j] = e;
    assert(i<j);
    assert(adjacency_size[i] < adjacency[i].size());
    assert(adjacency_size[j] < adjacency[j].size());
    adjacency(i, adjacency_size[i]++) = j;
    adjacency(j, adjacency_size[j]++) = i;
    }

    for(size_t i=0; i<adjacency.size(); ++i)
    std::sort(adjacency[i].begin(), adjacency[i].end());

    return adjacency;
    }
    */

    inline two_dim_variable_array<size_t> ILP_input::bipartite_variable_bdd_adjacency_matrix() const
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<std::pair<size_t,size_t>> var_bdd_adj_list;
        std::vector<size_t> degrees(this->nr_variables() + this->nr_constraints(), 0);

        // determine list of adjacencies and vertex degrees
        for(size_t i=0; i<this->linear_constraints_.size(); ++i) {
            const auto& l = this->linear_constraints_[i];
            for(const auto& v : l.variables) {
                var_bdd_adj_list.emplace_back(v.var, i);
                degrees[v.var]++;
                degrees[this->nr_variables() + i]++;
            }
        }

        // create adjacency matrix
        two_dim_variable_array<size_t> adjacency(degrees.begin(), degrees.end());
        std::fill(degrees.begin(), degrees.end(), 0);
        for (auto it = var_bdd_adj_list.begin(); it != var_bdd_adj_list.end(); it++)
        {
            size_t var = it->first;
            size_t bdd = this->nr_variables() + it->second;
            adjacency(var, degrees[var]++) = bdd;
            adjacency(bdd, degrees[bdd]++) = var;
        }

        return adjacency;
    }

    permutation ILP_input::reorder(ILP_input::variable_order var_ord)
    {
        if(var_ord == variable_order::input)
            return permutation(nr_variables());
        else if(var_ord == variable_order::bfs)
            return this->reorder_bfs();
        else if(var_ord == variable_order::cuthill)
            return this->reorder_Cuthill_McKee();
        else
        {
            assert(var_ord == variable_order::mindegree);
            return this->reorder_minimum_degree_ordering();
        }
    }

    void ILP_input::reorder(const permutation& order)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(order.size() == this->nr_variables());
        assert(order.is_permutation());
        std::vector<double> new_objective(this->nr_variables(), 0.0);
        std::vector<size_t> inverse_order(this->nr_variables());
        std::vector<std::string> new_var_index_to_name(this->nr_variables());

        for(size_t i=0; i<this->nr_variables(); ++i)
        {
            assert(var_name_to_index_.count(var_index_to_name_[order[i]]) > 0);
            var_name_to_index_.find(var_index_to_name_[order[i]]).value() = i;
        }

//#pragma omp parallel for schedule(guided)
        for(size_t i=0; i<this->nr_variables(); ++i)
        {
            if(order[i] < this->objective_.size())
                new_objective[i] = this->objective_[order[i]]; 

            if(order[i] < this->var_index_to_name_.size())
                new_var_index_to_name[i] = std::move(this->var_index_to_name_[order[i]]);

            inverse_order[order[i]] = i;

        }
        std::swap(this->objective_, new_objective);
        std::swap(new_var_index_to_name, this->var_index_to_name_);

//#pragma omp parallel for schedule(guided)
        for(size_t lc_index=0; lc_index<this->linear_constraints_.size(); ++lc_index)
        {
            auto& l = this->linear_constraints_[lc_index];
            for(auto& x : l.variables) {
                x.var = inverse_order[x.var];
            }
            l.normalize(); 
        }
    }

    inline permutation ILP_input::reorder_bfs()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = bfs_ordering(adj, this->nr_variables());
        reorder(order);
        return order;
    }

    inline permutation ILP_input::reorder_Cuthill_McKee()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = Cuthill_McKee(adj, this->nr_variables());
        reorder(order);
        return order;
    }

    inline permutation ILP_input::reorder_minimum_degree_ordering()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = minimum_degree_ordering(adj, this->nr_variables());
        reorder(order);
        return order;
    }

}
