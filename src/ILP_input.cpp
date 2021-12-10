#include "ILP_input.h"
#include <Eigen/Eigen>
#include "cuthill-mckee.h"
#include "bfs_ordering.hxx"
#include "minimum_degree_ordering.hxx"
#include "time_measure_util.h"
#include <iostream>

namespace LPMP { 
    bool ILP_input::constraint::monomials_cmp(const two_dim_variable_array<size_t>& monomials, const size_t idx1, const size_t idx2)
    {
        assert(monomials.size(idx1) > 0);
        assert(monomials.size(idx2) > 0);
        const size_t first_var1 = monomials(idx1,0);
        const size_t last_var1 = monomials(idx1,monomials.size(idx1)-1);
        const size_t first_var2 = monomials(idx2,0);
        const size_t last_var2 = monomials(idx2,monomials.size(idx2)-1);

        if(first_var1 != first_var2)
            return first_var1 < first_var2;
        else if(last_var1 != last_var2)
            return last_var1 < last_var2;
        else if(monomials.size(idx1) != monomials.size(idx2))
            return monomials.size(idx1) < monomials.size(idx2);
        else
            for(size_t i=0; i<monomials.size(idx1); ++i)
                if(monomials(idx1,i) != monomials(idx2,i))
                    return monomials(idx1,i) < monomials(idx2,i);
        return idx1 < idx2;
    }

    void ILP_input::constraint::normalize()
    {
        assert(coefficients.size() == monomials.size());
        for(size_t monomial_idx=0; monomial_idx<monomials.size(); ++monomial_idx)
        {
            assert(monomials.size(monomial_idx) > 0);
            std::sort(monomials.begin(monomial_idx), monomials.end(monomial_idx));
        }
        auto sort_pred = [&](const size_t idx1, const size_t idx2)
        {
            return monomials_cmp(monomials, idx1, idx2);
        };
        std::vector<size_t> idx(monomials.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), sort_pred);

        two_dim_variable_array<size_t> sorted_monomials;
        std::vector<int> sorted_coefficients;
        sorted_coefficients.reserve(coefficients.size());
        for(size_t i=0; i<idx.size(); ++i)
        {
            sorted_monomials.push_back(monomials.begin(idx[i]), monomials.end(idx[i]));
            sorted_coefficients.push_back(coefficients[idx[i]]);
        }

        std::swap(coefficients, sorted_coefficients);
        std::swap(monomials, sorted_monomials);
    }

    bool ILP_input::constraint::is_normalized() const
    {
        for(size_t i=0; i+1<monomials.size(); ++i)
            if(!monomials_cmp(monomials, i, i+1))
                return false;
        return true;
    }

    bool ILP_input::constraint::is_linear() const
    {
        for(size_t monomial_idx=0; monomial_idx<monomials.size(); ++monomial_idx)
            if(monomials.size(monomial_idx) != 1)
                return false;
        return true;
    }

    void ILP_input::normalize()
    {
#pragma omp parallel for
        for(size_t ineq_nr=0; ineq_nr<constraints_.size(); ++ineq_nr)
            constraints_[ineq_nr].normalize();
    }

    bool ILP_input::is_normalized() const
    {
        for(size_t ineq_nr=0; ineq_nr<constraints_.size(); ++ineq_nr)
            if(!constraints_[ineq_nr].is_normalized())
                return false;
        return true;
    }

    std::vector<size_t> ILP_input::variables(const size_t ineq_nr) const
    {
        assert(ineq_nr < constraints_.size());
        std::vector<size_t> vars;
        for(size_t monomial_idx=0; monomial_idx<constraints_[ineq_nr].monomials.size(); ++monomial_idx)
        {
            for(size_t var_idx=0; var_idx<constraints_[ineq_nr].monomials.size(monomial_idx); ++var_idx)
            {
                const size_t var = constraints_[ineq_nr].monomials(monomial_idx, var_idx);
                vars.push_back(var);
            }
        }
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
        return vars;
    }

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
        var_permutation_.push_back(var_permutation_.size()-1);
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
        {
            std::cout << var << ", " << nr_variables() << "\n";
            throw std::runtime_error("variable not present");
        }
        if(var >= objective_.size())
            return 0.0;
        return objective_[var];
    }

    double ILP_input::objective(const std::string& var) const
    {
        return objective(get_var_index(var));
    }

    size_t ILP_input::begin_new_inequality()
    {
        constraints_.push_back({});
        // set defaul inequality identifier
        std::string ineq_name = "INEQ_NR_" + std::to_string(constraints_.size()-1);
        set_inequality_identifier(ineq_name);
        return constraints_.size()-1;
    }

    void ILP_input::set_inequality_identifier(const std::string& identifier)
    {
        assert(constraints_.size() > 0);
        if(identifier == "")
            return;
        const std::string& old_identifier = constraints_.back().identifier;
        if(old_identifier != "")
        {
            assert(inequality_identifier_to_index_.count(old_identifier) > 0);
            inequality_identifier_to_index_.erase(old_identifier);
        }
        constraints_.back().identifier = identifier;
        if(inequality_identifier_to_index_.count(identifier) > 0)
            throw std::runtime_error("Duplicate inequality identifier " + identifier);
        inequality_identifier_to_index_.insert({identifier, constraints_.size()-1});
    }

    void ILP_input::set_inequality_type(const inequality_type ineq)
    {
        assert(constraints_.size() > 0);
        constraints_.back().ineq = ineq;
    }

    void ILP_input::add_to_constraint(const int coefficient, const size_t var)
    {
        assert(constraints_.size() > 0);
        auto& constr = constraints_.back();
        constr.coefficients.push_back(coefficient);
        std::array<size_t,1> vars{var};
        constr.monomials.push_back(vars.begin(), vars.end());
    }

    void ILP_input::add_to_constraint(const int coefficient, const std::string& var)
    {
        add_to_constraint(coefficient, get_or_create_variable_index(var));
    }

    void ILP_input::set_right_hand_side(const int x)
    {
        assert(constraints_.size() > 0);
        constraints_.back().right_hand_side = x;
    } 

    size_t ILP_input::nr_constraints() const
    {
        return constraints_.size();
    }

    bool ILP_input::preprocess()
    {
        for (auto it = constraints_.begin(); it != constraints_.end(); it++)
        {
            bool remove = false;

            // empty constraints
            if (it->monomials.size() == 0)
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
            if (it->monomials.size() == 1 && it->monomials.size(0) == 1)
            {
                assert(it->monomials.size() == it->coefficients.size());
                const size_t variable = it->monomials(0,0);
                const int coeff = it->coefficients[0];
                switch (it->ineq)
                {
                    case inequality_type::smaller_equal:
                        if(std::min(coeff, 0) > it->right_hand_side)
                            return false;
                        if(std::max(coeff, 0) <= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::greater_equal:
                        if(std::max(coeff, 0) < it->right_hand_side)
                            return false;
                        if(std::min(coeff, 0) >= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::equal:
                        if(it->right_hand_side != 0 && it->right_hand_side != coeff)
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
                *it = constraints_.back();
                constraints_.pop_back();
                it--;    
            }
        }

        return true;
    }

    inline two_dim_variable_array<size_t> ILP_input::variable_adjacency_matrix() const
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<Eigen::Triplet<int>> var_constraint_adjacency_list;

        for(size_t i=0; i<this->constraints_.size(); ++i) 
        {
            const auto vars = variables(i);
            for(const size_t var : vars)
                var_constraint_adjacency_list.push_back({var, i, 1});
        }

        Eigen::SparseMatrix<int> A(this->nr_variables(), this->constraints_.size());
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
        for(size_t i=0; i<this->constraints_.size(); ++i) 
        {
            const auto vars = variables(i);
            for(const size_t var : vars)
            {
                var_bdd_adj_list.emplace_back(var, i);
                degrees[var]++;
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
            var_permutation_ = permutation(nr_variables());
        else if(var_ord == variable_order::bfs)
            var_permutation_ = this->reorder_bfs();
        else if(var_ord == variable_order::cuthill)
            var_permutation_ = this->reorder_Cuthill_McKee();
        else
        {
            assert(var_ord == variable_order::mindegree);
            var_permutation_ = this->reorder_minimum_degree_ordering();
        }

        return var_permutation_;
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
        for(size_t ineq_nr=0; ineq_nr<this->constraints_.size(); ++ineq_nr)
        {
            auto& constr = this->constraints_[ineq_nr];
            for(size_t monomial_idx=0; monomial_idx<constr.monomials.size(); ++monomial_idx)
            {
                for(size_t var_idx=0; var_idx<constr.monomials.size(monomial_idx); ++var_idx)
                {
                    auto& var = constr.monomials(monomial_idx, var_idx);
                    var = inverse_order[var];
                }
            }
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

    size_t ILP_input::nr_constraint_groups() const
    {
        return coalesce_sets_.size(); 
    }

    std::tuple<const size_t*, const size_t*> ILP_input::constraint_group(const size_t i) const
    {
        assert(i < nr_constraint_groups());
        return std::make_tuple(coalesce_sets_[i].begin(), coalesce_sets_[i].end());
    }

    std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXi> ILP_input::export_constraints() const
    {
        using T = Eigen::Triplet<int>;
        std::vector<T> coefficients;

        for(size_t c=0; c<nr_constraints(); ++c)
        {
            for(size_t monomial_idx=0; monomial_idx<constraints()[c].monomials.size(); ++monomial_idx)
            {
                if(constraints()[c].monomials.size(monomial_idx) != 1)
                    throw std::runtime_error("instance has higher order constraints, cannot export to matrix");
                const size_t var = constraints()[c].monomials(monomial_idx, 0);
                const int coeff = constraints()[c].coefficients[monomial_idx];
                coefficients.push_back(T(c, var, coeff));

            }
        }

        Eigen::SparseMatrix<int> A(nr_constraints(), nr_variables());
        A.setFromTriplets(coefficients.begin(), coefficients.end());

        Eigen::MatrixXi b(nr_constraints(), 2);
        for(size_t c=0; c<nr_constraints(); ++c)
        {
            if(constraints()[c].ineq == inequality_type::equal)
            {
                b(c,0) = constraints()[c].right_hand_side;
                b(c,1) = constraints()[c].right_hand_side;
            }
            else if(constraints()[c].ineq == inequality_type::smaller_equal)
            {
                b(c,0) = std::numeric_limits<int>::min();
                b(c,1) = constraints()[c].right_hand_side;
            }
            else if(constraints()[c].ineq == inequality_type::greater_equal)
            {
                b(c,0) = constraints()[c].right_hand_side;
                b(c,1) = std::numeric_limits<int>::max();
            }
            else
            {
                assert(false);
            } 
        }

        return {A,b};
    }

}
