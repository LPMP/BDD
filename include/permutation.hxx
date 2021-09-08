#pragma once

#include <vector>
#include <numeric>

namespace LPMP {

    class permutation : public std::vector<size_t> {
        public:
            permutation(const size_t n);
            using std::vector<size_t>::vector;
            bool is_permutation() const;
            bool is_identity() const;
            permutation inverse_permutation() const;
            template<typename ITERATOR>
                std::vector<typename std::iterator_traits<ITERATOR>::value_type> permute(ITERATOR begin, ITERATOR end) const;
            template<typename ITERATOR>
                std::vector<typename std::iterator_traits<ITERATOR>::value_type> inverse_permute(ITERATOR begin, ITERATOR end) const;
    };

    inline permutation::permutation(const size_t n)
    {
        this->resize(n);
        std::iota(this->begin(), this->end(), 0);
    }

    template<typename ITERATOR>
        bool is_permutation(ITERATOR begin, ITERATOR end)
        {
            std::vector<char> nr_taken(std::distance(begin, end), 0);
            for(auto it=begin; it!=end; ++it) {
                if(*it >= std::distance(begin,end))
                    return false;
                if(nr_taken[*it] != 0)
                    return false;
                nr_taken[*it] = 1; 
            }
            return true;
        }

    template<typename ITERATOR>
        bool is_identity(ITERATOR begin, ITERATOR end)
        {
            for(size_t i=0; i<std::distance(begin,end); ++i)
                if(*(begin+i) != i)
                    return false;
            return true;
        }

    template<typename ITERATOR>
        permutation inverse_permutation(ITERATOR begin, ITERATOR end)
        {
            assert(is_permutation(begin, end));
            permutation inverse_perm(std::distance(begin, end));

            for(size_t i=0; i<std::distance(begin, end); ++i) 
                inverse_perm[*(begin+i)] = i;

            return inverse_perm;
        }

    inline bool permutation::is_permutation() const
    {
        return LPMP::is_permutation(this->begin(), this->end());
    }

    inline bool permutation::is_identity() const
    {
        return LPMP::is_identity(this->begin(), this->end());
    }

    inline permutation permutation::inverse_permutation() const
    {
        assert(is_permutation());
        return LPMP::inverse_permutation(this->begin(), this->end());
    }

    template<typename ITERATOR>
        std::vector<typename std::iterator_traits<ITERATOR>::value_type> permutation::permute(ITERATOR begin, ITERATOR end) const
        {
            assert(is_permutation());
            assert(std::distance(begin,end) == this->size());
            std::vector<typename std::iterator_traits<ITERATOR>::value_type> result;
            result.reserve(this->size());
            for(size_t i=0; i<this->size(); ++i) {
                result.push_back( *(begin+(*this)[i]) );
            }

            return result;
        }

    template<typename ITERATOR>
        std::vector<typename std::iterator_traits<ITERATOR>::value_type> permutation::inverse_permute(ITERATOR begin, ITERATOR end) const
        {
            assert(is_permutation());
            assert(std::distance(begin,end) == this->size());
            std::vector<typename std::iterator_traits<ITERATOR>::value_type> result(this->size());
            for(size_t i=0; i<this->size(); ++i)
                result[(*this)[i]] = *(begin+i);

            return result;
        }
}
