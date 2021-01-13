#pragma once

#include "bdd_base.hxx"

#include <cassert>
#include <vector>
#include <stack>

#include "bdd_variable.h"
#include "bdd_branch_node.h"

namespace LPMP {


    ////////////////////////////////////////////////////
    // Variable Fixing
    ////////////////////////////////////////////////////

    struct log_entry
    {
        log_entry(char * var_value)
        : var_value_(var_value) {}
        log_entry(bdd_branch_node_fix * source, bdd_branch_node_fix * target, bool high)
        : source_(source), target_(target), high_(high) {}

        void restore();

        char * var_value_ = nullptr;

        bdd_branch_node_fix * source_;
        bdd_branch_node_fix * target_;
        bool high_;
    };

    void log_entry::restore()
    {
        if (var_value_ != nullptr)
        {
            *var_value_ = 2;
            return;    
        }

        assert(target_ != bdd_branch_node_fix::terminal_0());
        if (high_)
        {
            assert(source_->high_outgoing == bdd_branch_node_fix::terminal_0());
            assert(source_->prev_high_incoming == nullptr);
            assert(source_->next_high_incoming == nullptr);
            source_->high_outgoing = target_;
            source_->bdd_var->nr_feasible_high_arcs++;
            if (bdd_branch_node_fix::is_terminal(target_))
                return;
            if (target_->first_high_incoming != nullptr)
                target_->first_high_incoming->prev_high_incoming = source_;
            source_->next_high_incoming = target_->first_high_incoming;
            target_->first_high_incoming = source_;
        }
        else
        {
            assert(source_->low_outgoing == bdd_branch_node_fix::terminal_0());
            assert(source_->prev_low_incoming == nullptr);
            assert(source_->next_low_incoming == nullptr);
            source_->low_outgoing = target_;
            source_->bdd_var->nr_feasible_low_arcs++;
            if (bdd_branch_node_fix::is_terminal(target_))
                return;
            if (target_->first_low_incoming != nullptr)
                target_->first_low_incoming->prev_low_incoming = source_;
            source_->next_low_incoming = target_->first_low_incoming;
            target_->first_low_incoming = source_;
        }
    }

    class bdd_fix_base : public bdd_base<bdd_variable_fix, bdd_branch_node_fix>
    {
        public:
            using bdd_base<bdd_variable_fix, bdd_branch_node_fix>::bdd_base;

            enum variable_order { marginals_absolute = 0, marginals_up = 1, marginals_down = 2, marginals_reduction = 3};
            enum variable_value { marginal = 0, reduction = 1, one = 2, zero = 3};

            bool fix_variables();

            bool fix_variables(const std::vector<size_t> & indices, const std::vector<char> & values);
            bool fix_variable(const size_t var, const char value);
            bool is_fixed(const size_t var) const;

            std::vector<double> search_space_reduction_coeffs();

            void count_forward_run(size_t last_var);
            void count_backward_run(ptrdiff_t first_var);

            void init_pointers();
            void set_total_min_marginals(const std::vector<double> total_min_marginals) { total_min_marginals_ = total_min_marginals; };
            void set_var_order(const variable_order var_order) { var_order_ = var_order; };
            void set_var_value(const variable_value var_value) { var_value_ = var_value; };

            void revert_changes(const size_t target_log_size);

            void init_primal_solution();
            const std::vector<char> & primal_solution() const { return primal_solution_; }
            const size_t log_size() const { return log_.size(); }

        private:
            bool remove_all_incoming_arcs(bdd_branch_node_fix & bdd_node);
            void remove_all_outgoing_arcs(bdd_branch_node_fix & bdd_node);
            void remove_outgoing_low_arc(bdd_branch_node_fix & bdd_node);
            void remove_outgoing_high_arc(bdd_branch_node_fix & bdd_node);

            std::vector<double> total_min_marginals_;
            variable_order var_order_;
            variable_value var_value_;
            std::vector<char> primal_solution_;
            std::stack<log_entry, std::deque<log_entry>> log_;
    };

    void bdd_fix_base::init_pointers()
    {
        for (size_t var = 0; var < this->nr_variables(); var++)
        {
            for (size_t bdd_index = 0; bdd_index < this->nr_bdds(var); bdd_index++)
            {
                auto & bdd_var = this->bdd_variables_(var, bdd_index);
                bdd_var.nr_feasible_low_arcs = 0;
                bdd_var.nr_feasible_high_arcs = 0;
                bdd_var.variable_index = var;
                for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
                {
                    auto & bdd_node = this->bdd_branch_nodes_[node_index];
                    if (bdd_node.low_outgoing != bdd_branch_node_fix::terminal_0())
                        bdd_var.nr_feasible_low_arcs++;
                    if (bdd_node.high_outgoing != bdd_branch_node_fix::terminal_0())
                        bdd_var.nr_feasible_high_arcs++;

                    bdd_node.bdd_var = & bdd_var;

                    auto * low_incoming = bdd_node.first_low_incoming;
                    while (low_incoming != nullptr && low_incoming->next_low_incoming != nullptr)
                    {
                        low_incoming->next_low_incoming->prev_low_incoming = low_incoming;
                        low_incoming = low_incoming->next_low_incoming;
                    }
                    auto * high_incoming = bdd_node.first_high_incoming;
                    while (high_incoming != nullptr && high_incoming->next_high_incoming != nullptr)
                    {
                        high_incoming->next_high_incoming->prev_high_incoming = high_incoming;
                        high_incoming = high_incoming->next_high_incoming;
                    }
                }
            }
        }
        init_primal_solution(); 
    }

    void bdd_fix_base::init_primal_solution()
    {
        primal_solution_.resize(this->nr_variables());
        std::fill(primal_solution_.begin(), primal_solution_.end(), 2);
    }

    bool bdd_fix_base::fix_variable(const std::size_t var, const char value)
    {
        assert(0 <= value && value <= 1);
        assert(primal_solution_.size() == this->nr_variables());
        assert(var < primal_solution_.size());

        // check if variable is already fixed
        if (primal_solution_[var] == value)
            return true;
        else if (is_fixed(var))
            return false;

        // mark variable as fixed
        primal_solution_[var] = value;
        const log_entry entry(&primal_solution_[var]);
        log_.push(entry);
        std::vector<std::pair<size_t, char>> restrictions;

        for (size_t bdd_index = 0; bdd_index < this->nr_bdds(var); bdd_index++)
        {
            auto & bdd_var = this->bdd_variables_(var, bdd_index);
            for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
            {
                auto & bdd_node = this->bdd_branch_nodes_[node_index];

                // skip isolated branch nodes
                if (bdd_node.is_first() && bdd_node.is_dead_end())
                    continue;

                if (value == 1)
                    remove_outgoing_low_arc(bdd_node);
                if (value == 0)
                    remove_outgoing_high_arc(bdd_node);

                // restructure parents if node is dead-end
                if (bdd_node.is_dead_end())
                {
                    if (!remove_all_incoming_arcs(bdd_node))
                        return false;
                }
            }

            // check if other variables in BDD are now restricted
            auto * cur = bdd_var.prev;
            while (cur != nullptr)
            {
                if (is_fixed(cur->variable_index))
                {
                    cur = cur->prev;
                    continue;
                }
                if (cur->nr_feasible_low_arcs == 0)
                    restrictions.emplace_back(cur->variable_index, 1);
                if (cur->nr_feasible_high_arcs == 0)
                    restrictions.emplace_back(cur->variable_index, 0);
                cur = cur->prev;
            }
            cur = bdd_var.next;
            while (cur != nullptr)
            {
                if (is_fixed(cur->variable_index))
                {
                    cur = cur->next;
                    continue;
                }
                if (cur->nr_feasible_low_arcs == 0)
                    restrictions.emplace_back(cur->variable_index, 1);
                if (cur->nr_feasible_high_arcs == 0)
                    restrictions.emplace_back(cur->variable_index, 0);
                cur = cur->next;
            }
        }

        // fix implied restrictions
        for (auto & restriction : restrictions)
        {
            if (!fix_variable(restriction.first, restriction.second))
                return false;
        }

        return true;
    }

    bool bdd_fix_base::remove_all_incoming_arcs(bdd_branch_node_fix & bdd_node)
    {
        if (bdd_node.is_first())
            return false;
        // low arcs
        {
            auto * cur = bdd_node.first_low_incoming;
            while (cur != nullptr)
            {
                // log change
                assert(cur->low_outgoing == &bdd_node);
                auto * temp = cur;
                const log_entry entry(cur, &bdd_node, false);
                log_.push(entry);
                // remove arc
                cur->low_outgoing = bdd_branch_node_fix::terminal_0();
                assert(cur->bdd_var != nullptr);
                cur->bdd_var->nr_feasible_low_arcs--;
                bdd_node.first_low_incoming = cur->next_low_incoming;
                cur = cur->next_low_incoming;
                // remove list pointers
                if (cur != nullptr)
                {
                    assert(cur->prev_low_incoming != nullptr);
                    cur->prev_low_incoming->next_low_incoming = nullptr;
                    cur->prev_low_incoming = nullptr;    
                }
                // recursive call if parent is dead-end
                if (temp->is_dead_end())
                {
                    if (!remove_all_incoming_arcs(*temp))
                        return false;
                }
            }
        }
        // high arcs
        {
            auto * cur = bdd_node.first_high_incoming;
            while (cur != nullptr)
            {
                assert(cur->high_outgoing == &bdd_node);
                auto * temp = cur;
                const log_entry entry(cur, &bdd_node, true);
                log_.push(entry);
                cur->high_outgoing = bdd_branch_node_fix::terminal_0();
                assert(cur->bdd_var != nullptr);
                cur->bdd_var->nr_feasible_high_arcs--;
                bdd_node.first_high_incoming = cur->next_high_incoming; 
                cur = cur->next_high_incoming; 
                if (cur != nullptr)
                {
                    assert(cur->prev_low_incoming != nullptr);
                    cur->prev_high_incoming->next_high_incoming = nullptr;
                    cur->prev_high_incoming = nullptr;    
                }
                if (temp->is_dead_end())
                {
                    if (!remove_all_incoming_arcs(*temp))
                        return false;
                }
            }      
        }
        return true;
    }

    void bdd_fix_base::remove_all_outgoing_arcs(bdd_branch_node_fix & bdd_node)
    {
        remove_outgoing_low_arc(bdd_node);
        remove_outgoing_high_arc(bdd_node);
    }

    void bdd_fix_base::remove_outgoing_low_arc(bdd_branch_node_fix & bdd_node)
    {
        if (!bdd_branch_node_fix::is_terminal(bdd_node.low_outgoing))
        {
            // change pointers
            if (bdd_node.prev_low_incoming == nullptr)
                bdd_node.low_outgoing->first_low_incoming = bdd_node.next_low_incoming;
            else
                bdd_node.prev_low_incoming->next_low_incoming = bdd_node.next_low_incoming;
            if (bdd_node.next_low_incoming != nullptr)
                bdd_node.next_low_incoming->prev_low_incoming = bdd_node.prev_low_incoming;
            bdd_node.prev_low_incoming = nullptr;
            bdd_node.next_low_incoming = nullptr;
            // recursive call if child node is unreachable
            if (bdd_node.low_outgoing->is_first())
                remove_all_outgoing_arcs(*bdd_node.low_outgoing);
        }
        if (bdd_node.low_outgoing != bdd_branch_node_fix::terminal_0())
        {
            // log change
            const log_entry entry(&bdd_node, bdd_node.low_outgoing, false);
            log_.push(entry);
            // remove arc
            bdd_node.low_outgoing = bdd_branch_node_fix::terminal_0();
            bdd_node.bdd_var->nr_feasible_low_arcs--;
        } 
    }

    void bdd_fix_base::remove_outgoing_high_arc(bdd_branch_node_fix & bdd_node)
    {
        if (!bdd_branch_node_fix::is_terminal(bdd_node.high_outgoing))
        {
            if (bdd_node.prev_high_incoming == nullptr)
                bdd_node.high_outgoing->first_high_incoming = bdd_node.next_high_incoming;
            else
                bdd_node.prev_high_incoming->next_high_incoming = bdd_node.next_high_incoming;
            if (bdd_node.next_high_incoming != nullptr)
                bdd_node.next_high_incoming->prev_high_incoming = bdd_node.prev_high_incoming;
            bdd_node.prev_high_incoming = nullptr;
            bdd_node.next_high_incoming = nullptr;
            if (bdd_node.high_outgoing->is_first())
                remove_all_outgoing_arcs(*bdd_node.high_outgoing);
        }
        if (bdd_node.high_outgoing != bdd_branch_node_fix::terminal_0())
        {
            const log_entry entry(&bdd_node, bdd_node.high_outgoing, true);
            log_.push(entry);
            bdd_node.high_outgoing = bdd_branch_node_fix::terminal_0();
            bdd_node.bdd_var->nr_feasible_high_arcs--;
        }
    }

    bool bdd_fix_base::fix_variables(const std::vector<size_t> & variables, const std::vector<char> & values)
    {
        assert(variables.size() == values.size());

        init_primal_solution();

        struct VarFix
        {
            VarFix(const size_t log_size, const size_t index, const char val)
            : log_size_(log_size), index_(index), val_(val) {}
            
            const size_t log_size_;
            const size_t index_;
            const char val_;
        };

        std::stack<VarFix, std::deque<VarFix>> variable_fixes;
        variable_fixes.emplace(log_.size(), 0, 1-values[0]);
        variable_fixes.emplace(log_.size(), 0, values[0]);

        size_t nfixes = 0;
        size_t max_fixes = this->nr_variables();
        // size_t max_fixes = std::numeric_limits<size_t>::max();
        std::cout << "Search tree node budget: " << max_fixes << std::endl;
        std::cout << "Searching for feasible solution..." << std::endl;

        while (!variable_fixes.empty())
        {
            nfixes++;
            if (nfixes > max_fixes)
            {
                std::cout << "No feasible solution found within budget." << std::endl;
                return false;
            }

            auto fix = variable_fixes.top();
            variable_fixes.pop();
            size_t index = fix.index_;

            revert_changes(fix.log_size_);
            bool feasible = fix_variable(variables[index], fix.val_);

            if (!feasible)
                continue;

            while (is_fixed(variables[index]))
            {
                index++;
                if (index >= variables.size())
                {
                    std::cout << "Found feasible solution after expanding " << nfixes << " search tree nodes." << std::endl;
                    return true;
                }
            }

            variable_fixes.emplace(log_.size(), index, 1-values[index]);
            variable_fixes.emplace(log_.size(), index, values[index]);
        }

        std::cout << "Expanded " << nfixes << " search tree nodes." << std::endl;
        std::cout << "Problem appears to be infeasible." << std::endl;
        return false;
    }

    bool bdd_fix_base::is_fixed(const size_t var) const
    {
        assert(primal_solution_.size() == this->nr_variables());
        assert(var < primal_solution_.size());
        return primal_solution_[var] < 2;
    }

    void bdd_fix_base::revert_changes(const size_t target_log_size)
    {
        while (log_.size() > target_log_size)
        {
            log_.top().restore();
            log_.pop();
        }
    }

    void bdd_fix_base::count_backward_run(ptrdiff_t first_var)
    {
        assert(first_var >= 0 && first_var < this->nr_variables());
        for (ptrdiff_t var = this->nr_variables()-1; var >= first_var; --var)
        {
            for (size_t bdd_index=0; bdd_index < this->nr_bdds(var); bdd_index++)
            {
                auto & bdd_var = this->bdd_variables_(var, bdd_index);
                for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
                {
                    auto & bdd_node = this->bdd_branch_nodes_[node_index];
                    bdd_node.count_backward_step();
                }
            }
        } 
    }

    void bdd_fix_base::count_forward_run(size_t last_var)
    {
        assert(last_var >= 0 && last_var < this->nr_variables());
        for (size_t var = 0; var <= last_var; var++)
        {
            for (size_t bdd_index=0; bdd_index < this->nr_bdds(var); bdd_index++)
            {
                auto & bdd_var = this->bdd_variables_(var, bdd_index);
                for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
                {
                    auto & bdd_node = this->bdd_branch_nodes_[node_index];
                    bdd_node.count_forward_step();
                }
            }
        }
    }

    std::vector<double> bdd_fix_base::search_space_reduction_coeffs()
    {
        std::vector<double> r_coeffs;
        // solution count backward run
        count_backward_run(0);
        // solution count forward run
        for (size_t var = 0; var < this->nr_variables(); var++)
        {
            double coeff = 0;
            for (size_t bdd_index=0; bdd_index < this->nr_bdds(var); bdd_index++)
            {
                auto & bdd_var = this->bdd_variables_(var, bdd_index);
                for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
                {
                    auto & bdd_node = this->bdd_branch_nodes_[node_index];
                    bdd_node.count_forward_step();
                    coeff += bdd_node.count_high() - bdd_node.count_low();
                }
            }
            r_coeffs.push_back(coeff);
        }

        return r_coeffs;
    }

    bool bdd_fix_base::fix_variables()
    {
        std::vector<double> reduction_coeffs = search_space_reduction_coeffs();
        std::vector<size_t> variables;
        for (size_t i = 0; i < this->nr_variables(); i++)
            variables.push_back(i);

        const double eps = std::numeric_limits<double>::epsilon();

        auto sign = [](const double val) -> double
        {
            if (val < 0)
                return -1.0;
            else if (val > 0)
                return 1.0;
            else
                return 0.0;
        };

        auto order_reduction = [&](const size_t a, const size_t b)
        {
            return sign(reduction_coeffs[a]) * total_min_marginals_[a] > sign(reduction_coeffs[b]) * total_min_marginals_[b];
        };

        auto order_abs = [&](const size_t a, const size_t b)
        {
            return std::abs(total_min_marginals_[a]) > std::abs(total_min_marginals_[b]);
        };

        auto order_up = [&](const size_t a, const size_t b)
        {
            return total_min_marginals_[a] < total_min_marginals_[b];
        };

        auto order_up_down = [&](const size_t a, const size_t b)
        {
            if (total_min_marginals_[a] > eps && total_min_marginals_[b] > eps)
                return total_min_marginals_[a] > total_min_marginals_[b];
            return total_min_marginals_[a] < total_min_marginals_[b];
        };

        auto order_down_up = [&](const size_t a, const size_t b)
        {
            if (total_min_marginals_[a] < -eps && total_min_marginals_[b] < -eps)
                return total_min_marginals_[a] < total_min_marginals_[b];
            return total_min_marginals_[a] > total_min_marginals_[b];
        };

        auto order_down = [&](const size_t a, const size_t b)
        {
            return total_min_marginals_[a] > total_min_marginals_[b];
        };

        if (var_order_ == variable_order::marginals_absolute)
            std::sort(variables.begin(), variables.end(), order_abs);
        else if (var_order_ == variable_order::marginals_up)
            std::sort(variables.begin(), variables.end(), order_up);
        else if (var_order_ == variable_order::marginals_down)
            std::sort(variables.begin(), variables.end(), order_down);
        else if (var_order_ == variable_order::marginals_reduction)
            std::sort(variables.begin(), variables.end(), order_reduction);
        else
            std::sort(variables.begin(), variables.end(), order_up);

        std::vector<char> values;
        for (size_t i = 0; i < variables.size(); i++)
        {
            char val;
            if (var_value_ == variable_value::marginal)
                val = (total_min_marginals_[variables[i]] < eps) ? 1 : 0;
            else if (var_value_ == variable_value::reduction)
                val = (sign(reduction_coeffs[i]) < 0) ? 1 : 0;
            else if (var_value_ == variable_value::one)
                val = 1;
            else if (var_value_ == variable_value::zero)
                val = 0;
            else
                val = (total_min_marginals_[variables[i]] < eps) ? 1 : 0;
            values.push_back(val);
        }

        return fix_variables(variables, values);
    }

    
}
