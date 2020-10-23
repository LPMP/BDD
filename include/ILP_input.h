#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cassert>
#include <algorithm>
#include "two_dimensional_variable_array.hxx"
#include "permutation.hxx"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

namespace LPMP {

    class ILP_input {
        public:
        struct weighted_variable {
            int coefficient;
            std::size_t var;
            bool operator<(const weighted_variable& o) const { return var < o.var; }
        };

        enum class inequality_type { smaller_equal, greater_equal, equal };

        struct linear_constraint {
            std::vector<weighted_variable> variables;
            inequality_type ineq;
            int right_hand_side;
            void normalize() { std::sort(variables.begin(), variables.end()); }
        };

        enum class variable_order { input, bfs, cuthill, mindegree } variable_order_{variable_order::input};

        bool var_exists(const std::string& var) const;
        std::size_t get_var_index(const std::string& var) const;
        std::string get_var_name(const std::size_t index) const;
        std::size_t add_new_variable(const std::string& var);
        std::size_t get_or_create_variable_index(const std::string& var);
        std::size_t nr_variables() const;
        void add_to_objective(const double coefficient, const std::string& var);
        void add_to_objective(const double coefficient, const std::size_t var);
        const std::vector<double>& objective() const { return objective_; };
        double objective(const std::size_t var) const;
        double objective(const std::string& var) const;
        void begin_new_inequality();
        void set_inequality_type(const inequality_type ineq);
        void add_to_constraint(const int coefficient, const std::size_t var);
        void add_to_constraint(const int coefficient, const std::string& var);
        void set_right_hand_side(const int x);
        std::size_t nr_constraints() const;
        const auto& constraints() const { return linear_constraints_; }

        template<typename ITERATOR>
            bool check_feasibility(ITERATOR begin, ITERATOR end) const;

        template<typename ITERATOR>
            double evaluate(ITERATOR begin, ITERATOR end) const; 

        template<typename STREAM>
            void write(STREAM& s) const;

        permutation reorder(variable_order var_ord);
        permutation reorder_bfs();
        permutation reorder_Cuthill_McKee(); 
        permutation reorder_minimum_degree_averaging();
        void reorder(const permutation& new_order);

        private:
            std::vector<linear_constraint> linear_constraints_;
            std::vector<double> objective_;
            std::vector<std::string> var_index_to_name_;
            //std::unordered_map<std::string, std::size_t> var_name_to_index_;
            tsl::robin_map<std::string, std::size_t> var_name_to_index_;

        private:
            two_dim_variable_array<std::size_t> variable_adjacency_matrix() const;
    };

    template<typename ITERATOR>
        bool ILP_input::check_feasibility(ITERATOR begin, ITERATOR end) const
        {
            if(std::distance(begin, end) != nr_variables())
                return false;

            for(const auto& l : linear_constraints_) {
                int s = 0;
                for(const auto v : l.variables) {
                    assert(*(begin + v.var) == 0 || *(begin + v.var) == 1);
                    s += v.coefficient * *(begin + v.var);
                }
                switch(l.ineq) {
                    case inequality_type::smaller_equal:
                        if(s > l.right_hand_side)
                            return false;
                        break;
                    case inequality_type::greater_equal:
                        if(s < l.right_hand_side)
                            return false;
                        break;
                    case inequality_type::equal:
                        if(s != l.right_hand_side)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
            }
            return true;
        }

    template<typename ITERATOR>
        double ILP_input::evaluate(ITERATOR begin, ITERATOR end) const
        {
            if(!check_feasibility(begin,end))
                return std::numeric_limits<double>::infinity();
            assert(std::distance(begin,end) >= objective_.size());
            double cost = 0.0;
            for(std::size_t i=0; i<objective_.size(); ++i) {
                assert(*(begin+i) == 0 || *(begin+i) == 1);
                cost += objective_[i] * *(begin+i);
            }
            return cost;
        }

    template<typename STREAM>
        void ILP_input::write(STREAM& s) const
        {
            s << "Minimize\n";
            for(const auto o : var_name_to_index_) {
                s << (objective(o.second) < 0.0 ? "- " : "+ ") <<  std::abs(objective(o.second)) << " " << o.first << "\n"; 
            }
            s << "Subject To\n";
            for(const auto& ineq : constraints()) {
                for(const auto term : ineq.variables) {
                    s << (term.coefficient < 0.0 ? "- " : "+ ") <<  std::abs(term.coefficient) << " " << var_index_to_name_[term.var] << " "; 
                }

                switch(ineq.ineq) {
                    case inequality_type::smaller_equal:
                        s << " <= ";
                        break;
                    case inequality_type::greater_equal:
                        s << " >= ";
                        break;
                    case inequality_type::equal:
                        s << " = ";
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                        break;
                }
                s << ineq.right_hand_side << "\n";
            }
            s << "Bounds\n";
            s << "Binaries\n";
            for(const auto& v : var_index_to_name_)
                s << v << "\n";
            s << "End\n";
        } 
}
