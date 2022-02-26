#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <limits>

namespace LPMP {

    // general two-dimensional array with variable first and second dimension sizes, i.e. like vector<vector<T>>. Holds all data contiguously and therefore may be more efficient than vector<vector<T>>

    // do zrobienia: - alignment?
    //               - custom memory allocator?
    //               - iterators
    template<typename T>
        class two_dim_variable_array
        {
            public:
                template<class I> friend class two_dim_variable_array;

                two_dim_variable_array();
                ~two_dim_variable_array();

                template<typename I>
                    two_dim_variable_array(const two_dim_variable_array<I>& o);

                two_dim_variable_array(const two_dim_variable_array<T>& o);

                // iterator holds size of each dimension of the two dimensional array
                template<typename I>
                    two_dim_variable_array(const std::vector<I>& size);

                two_dim_variable_array(const std::vector<std::vector<T>>& o);

                // iterator holds size of each dimension of the two dimensional array
                template<typename ITERATOR>
                    two_dim_variable_array(ITERATOR size_begin, ITERATOR size_end);

                // iterator holds size of each dimension of the two dimensional array
                template<typename ITERATOR>
                    two_dim_variable_array(ITERATOR size_begin, ITERATOR size_end, const T& val);

                void clear();
                template<typename ITERATOR>
                    void push_back(ITERATOR val_begin, ITERATOR val_end);

                template<typename ITERATOR>
                    void resize(ITERATOR begin, ITERATOR end);

                template<typename ITERATOR>
                    void resize(ITERATOR begin, ITERATOR end, T val);

                struct ConstArrayAccessObject
                {
                    ConstArrayAccessObject(const T* begin, const T* end) : begin_(begin), end_(end) { assert(begin <= end); }
                    const T& operator[](const size_t i) const { assert(i < size()); return begin_[i]; }
                    size_t size() const {  return (end_ - begin_); }

                    const T* begin() { return begin_; }
                    const T* end() { return end_; }
                    auto rbegin() { return std::make_reverse_iterator(end()); }
                    auto rend() { return std::make_reverse_iterator(begin()); }
                    const T& back() { return *(end()-1); }

                    const T* begin() const { return begin_; }
                    const T* end() const { return end_; }
                    auto rbegin() const { return std::make_reverse_iterator(end()); }
                    auto rend() const { return std::make_reverse_iterator(begin()); }
                    const T& back() const { return *(end()-1); }

                    private:
                    const T* begin_;
                    const T* end_;
                };
                struct ArrayAccessObject
                {
                    ArrayAccessObject(T* begin, T* end) : begin_(begin), end_(end) { assert(begin <= end); }
                    template<typename VEC>
                        void operator=(const VEC& o) 
                        { 
                            assert(o.size() == this->size());
                            const auto s = this->size();
                            for(size_t i=0; i<s; ++i) {
                                (*this)[i] = o[i];
                            }
                        }
                    const T& operator[](const size_t i) const { assert(i < size()); return begin_[i]; }
                    T& operator[](const size_t i) { assert(i < size()); return begin_[i]; }
                    size_t size() const {  return (end_ - begin_); }

                    T* begin() { return begin_; }
                    T* end() { return end_; }
                    auto rbegin() { return std::make_reverse_iterator(end()); }
                    auto rend() { return std::make_reverse_iterator(begin()); }
                    T& back() { return *(end()-1); }

                    T const* begin() const { return begin_; }
                    T const* end() const { return end_; }
                    auto rbegin() const { return std::make_reverse_iterator(end()); }
                    auto rend() const { return std::make_reverse_iterator(begin()); }
                    const T& back() const { return *(end()-1); }

                    private:
                    T* begin_;
                    T* end_;
                };
                ArrayAccessObject operator[](const size_t i);
                ConstArrayAccessObject operator[](const size_t i) const;
                const T& operator()(const size_t i, const size_t j) const;
                T& operator()(const size_t i, const size_t j);

                size_t nr_elements() const;
                size_t size() const;
                size_t size(const size_t i) const;

                ConstArrayAccessObject back() const;
                ArrayAccessObject back();

                struct iterator : public std::iterator< std::random_access_iterator_tag, T* > {
                    iterator(T* _data, size_t* _offset) : data(_data), offset(_offset) {}
                    void operator++() { ++offset; }
                    void operator--() { --offset; }
                    iterator& operator+=(const size_t i) { offset+=i; return *this; }
                    iterator& operator-=(const size_t i) { offset-=i; return *this; }
                    iterator operator+(const size_t i) { iterator it(data,offset+i); return it; }
                    iterator operator-(const size_t i) { iterator it(data,offset-i); return it; }
                    auto operator-(const iterator it) const { return offset - it.offset; }
                    ArrayAccessObject operator*() { return ArrayAccessObject(data+*offset,data+*(offset+1)); }
                    const ArrayAccessObject operator*() const { return ArrayAccessObject(data+*offset,data+*(offset+1)); }
                    bool operator==(const iterator it) const { return data == it.data && offset == it.offset; }
                    bool operator!=(const iterator it) const { return !(*this == it); }
                    T* data;
                    size_t* offset;
                };

                struct reverse_iterator : public std::iterator_traits< T* > {
                    reverse_iterator(T* _data, size_t* _offset) : data(_data), offset(_offset) {}
                    void operator++() { --offset; }
                    void operator--() { ++offset; }
                    reverse_iterator& operator+=(const size_t i) { offset-=i; return *this; }
                    reverse_iterator& operator-=(const size_t i) { offset+=i; return *this; }
                    iterator operator+(const size_t i) { iterator it(data,offset-i); return it; }
                    iterator operator-(const size_t i) { iterator it(data,offset+i); return it; }
                    auto operator-(const reverse_iterator it) const { return it.offset - offset; }
                    ArrayAccessObject operator*() { return ArrayAccessObject(data+*(offset-1),data+*offset); }
                    const ArrayAccessObject operator*() const { return ArrayAccessObject(data+*(offset-1),data+*offset); }
                    bool operator==(const reverse_iterator it) const { return data == it.data && offset == it.offset; }
                    bool operator!=(const reverse_iterator it) const { return !(*this == it); }
                    T* data;
                    size_t* offset;
                };

                iterator begin();
                iterator end();

                // TODO: correct?
                iterator begin() const;
                iterator end() const;

                reverse_iterator rbegin();
                reverse_iterator rend();

                struct size_iterator : public std::iterator_traits<const size_t*> {
                    size_iterator(const size_t* _offset) : offset(_offset) {}
                    size_t operator*() const { return *(offset+1) - *offset; } 
                    void operator++() { ++offset; }
                    auto operator-(const size_iterator it) const { return offset - it.offset; }
                    size_iterator operator-(const size_t i) const { size_iterator it(offset-i); return it; }
                    bool operator==(const size_iterator it) const { return offset == it.offset; }
                    bool operator!=(const size_iterator it) const { return !(*this == it); }
                    const size_t* offset;
                };

                size_iterator size_begin() const;
                size_iterator size_end() const;

                T* begin(const size_t idx);
                T* end(const size_t idx);

                void set(const T& val);

                std::vector<T>& data();
                const std::vector<T>& data() const;

                size_t first_index(const T* p) const;
                std::array<size_t,2> indices(const T* p) const;
                size_t index(const size_t first_index, const size_t second_index) const;

            private:
                template<typename ITERATOR>
                    size_t set_dimensions(ITERATOR begin, ITERATOR end);

                template<typename ITERATOR>
                    static std::vector<size_t> compute_offsets(ITERATOR begin, ITERATOR end);

                std::vector<size_t> offsets_;
                std::vector<T> data_;
                // memory is laid out like this:
                // pointer[1], pointer[2], ... , pointer[dim1], pointer[end], data[1], data[2], ..., data[dim1] .
                // first pointers to the arrays in the second dimension are stored, then contiguously the data itself.
        };

    template<typename T>
        two_dim_variable_array<T>::two_dim_variable_array() 
        : two_dim_variable_array(std::vector<size_t>{})
        {}

    template<typename T>
        two_dim_variable_array<T>::~two_dim_variable_array() 
        {
            static_assert(!std::is_same_v<T,bool>, "value type cannot be bool");
        }

    template<typename T>
        template<typename I>
        two_dim_variable_array<T>::two_dim_variable_array(const two_dim_variable_array<I>& o)
        : offsets_(o.offsets_),
        data_(offsets_.back())
    {
        // possible check if types are convertible and if so, convert
    }

    template<typename T>
        two_dim_variable_array<T>::two_dim_variable_array(const two_dim_variable_array<T>& o)
        : offsets_(o.offsets_),
        data_(o.data_)
    {}

    // iterator holds size of each dimension of the two dimensional array
    template<typename T>
        template<typename I>
        two_dim_variable_array<T>::two_dim_variable_array(const std::vector<I>& size)
        : offsets_(compute_offsets(size.begin(), size.end())),
        data_(offsets_.back())
    {
        //const size_t s = set_dimensions(size.begin(), size.end());
        //data_.resize(s);
    }

    template<typename T>
        two_dim_variable_array<T>::two_dim_variable_array(const std::vector<std::vector<T>>& o)
        {
            std::vector<size_t> size_array;
            size_array.reserve(o.size());
            for(const auto& v : o) {
                size_array.push_back(v.size());
            }
            const size_t s = set_dimensions(size_array.begin(), size_array.end());
            data_.resize(s);

            for(size_t i=0; i<size(); ++i) {
                for(size_t j=0; j<(*this)[i].size(); ++j) {
                    (*this)(i,j) = o[i][j];
                }
            }
        }

    // iterator holds size of each dimension of the two dimensional array
    template<typename T>
        template<typename ITERATOR>
        two_dim_variable_array<T>::two_dim_variable_array(ITERATOR size_begin, ITERATOR size_end)
        : offsets_(compute_offsets(size_begin, size_end)),
        data_(offsets_.back())
    {
        //const size_t s = set_dimensions(size_begin, size_end);
        //data_.resize(s);
    }

    // iterator holds size of each dimension of the two dimensional array
    template<typename T>
        template<typename ITERATOR>
        two_dim_variable_array<T>::two_dim_variable_array(ITERATOR size_begin, ITERATOR size_end, const T& val)
        : offsets_(compute_offsets(size_begin, size_end)),
        data_(offsets_.back(), val)
    {
        //const size_t s = set_dimensions(size_begin, size_end);
        //data_.resize(s);
    }

    /*
       friend std::ostream& operator<<(std::ostream& os, const two_dim_variable_array<T>& a) {
       for(size_t i=0; i<a.size(); ++i) {
       for(size_t j=0; j<a[i].size(); ++j) {
       os << a(i,j) << " ";
       }
       os << "\n";
       }
       return os;
       }
       */

    template<typename T>
        template<typename ITERATOR>
        void two_dim_variable_array<T>::resize(ITERATOR begin, ITERATOR end)
        {
            const size_t size = set_dimensions(begin,end);
            data_.resize(size); 
        }

    template<typename T>
        template<typename ITERATOR>
        void two_dim_variable_array<T>::resize(ITERATOR begin, ITERATOR end, T val)
        {
            const size_t size = set_dimensions(begin,end);
            data_.resize(size, val); 
        }

    template<typename T>
        typename two_dim_variable_array<T>::ArrayAccessObject two_dim_variable_array<T>::operator[](const size_t i) 
        {
            assert(i<this->size());
            return ArrayAccessObject( &data_[offsets_[i]], &data_[offsets_[i+1]] );
        }

    template<typename T>
        typename two_dim_variable_array<T>::ConstArrayAccessObject two_dim_variable_array<T>::operator[](const size_t i) const 
        {
            assert(i<this->size());
            return ConstArrayAccessObject( &data_[offsets_[i]], &data_[offsets_[i+1]] );
        }

    template<typename T>
        const T& two_dim_variable_array<T>::operator()(const size_t i, const size_t j) const
        {
            assert(i<size() && j< (*this)[i].size());
            return data_[offsets_[i]+j];
        }

    template<typename T>
        T& two_dim_variable_array<T>::operator()(const size_t i, const size_t j)
        {
            assert(i < size() && j< (*this)[i].size());
            return data_[offsets_[i]+j];
        }

    template<typename T>
        size_t two_dim_variable_array<T>::nr_elements() const { return data_.size(); }

    template<typename T>
        size_t two_dim_variable_array<T>::size() const { assert(offsets_.size() > 0); return offsets_.size()-1; }

    template<typename T>
        size_t two_dim_variable_array<T>::size(const size_t i) const { assert(i < size()); return offsets_[i+1] - offsets_[i]; }

    template<typename T>
        typename two_dim_variable_array<T>::ConstArrayAccessObject two_dim_variable_array<T>::back() const 
        {
            const size_t i = size()-1;
            return (*this)[i];
        }

    template<typename T>
        typename two_dim_variable_array<T>::ArrayAccessObject two_dim_variable_array<T>::back()
        {
            const size_t i = size()-1;
            return (*this)[i];
        }

    template<typename T>
        T* two_dim_variable_array<T>::begin(const size_t idx) 
        { 
            assert(idx < size());
            return &data_[offsets_[idx]];
        }

    template<typename T>
        T* two_dim_variable_array<T>::end(const size_t idx)
        {
            assert(idx < size());
            return &data_[offsets_[idx+1]];
        }

    template<typename T>
        void two_dim_variable_array<T>::set(const T& val)
        {
            for(size_t i=0; i<size(); ++i) {
                for(size_t j=0; j<(*this)[i].size(); ++j) {
                    (*this)(i,j) = val;
                }
            }
        }

    template<typename T>
        std::vector<T>& two_dim_variable_array<T>::data() 
        {
            return data_;
        }

    template<typename T>
        const std::vector<T>& two_dim_variable_array<T>::data() const 
        {
            return data_; 
        }

    template<typename T>
        size_t two_dim_variable_array<T>::first_index(const T* p) const
        {
            assert(false); // TODO: does not work yet, write test
            assert(p >= &data_[0]);
            assert(p < &data_[0] + data_.size());

            // binary search to locate first index
            size_t lower = 0;
            size_t upper = size()-1;
            while(lower <= upper) {
                const size_t middle = (lower+upper)/2;
                if( &(*this)(middle,0) <= p && p <= &(*this)[middle].back()) {
                    return middle; 
                } else if( &(*this)(middle,0) < p ) {
                    lower = middle+1;
                } else {
                    upper = middle-1; 
                }
            }
            assert(false);
            return std::numeric_limits<size_t>::max();
        }

    template<typename T>
        std::array<size_t,2> two_dim_variable_array<T>::indices(const T* p) const
        {
            assert(p >= &data_[0]);
            assert(p < &data_[0] + data_.size());

            const size_t first_idx = first_index(p);

            // binary search for second index
            const size_t second_index = [&]() {
                size_t lower = 0;
                size_t upper = (*this)[first_idx].size()-1;
                while(lower <= upper) {
                    const size_t middle = (lower+upper)/2;
                    if(&(*this)(first_idx,middle) < p)
                        lower = middle+1; 
                    else if(&(*this)(first_idx,middle) > p)
                        upper = middle-1;
                    else
                        return middle;
                }
                assert(false);
                return std::numeric_limits<size_t>::max();
            }();

            assert(p == &(*this)(first_idx, second_index));
            return {first_idx, second_index};
        }

    template<typename T>
        size_t two_dim_variable_array<T>::index(const size_t first_index, const size_t second_index) const
        {
            assert(first_index < size());
            assert(second_index < (*this)[first_index].size());
            return std::distance(&data_[0], &(*this)(first_index, second_index)); 
        }

    template<typename T>
        template<typename ITERATOR>
        size_t two_dim_variable_array<T>::set_dimensions(ITERATOR begin, ITERATOR end)
        {
            // first calculate amount of memory needed in bytes
            const auto s = std::distance(begin, end);
            offsets_.clear();
            offsets_.reserve(s);
            offsets_.push_back(0);
            for(auto it=begin; it!=end; ++it) {
                assert(*it >= 0);
                offsets_.push_back( offsets_.back() + *it );
            }
            return offsets_.back();
        }

    template<typename T>
        template<typename ITERATOR>
        std::vector<size_t> two_dim_variable_array<T>::compute_offsets(ITERATOR begin, ITERATOR end)
        {
            // first calculate amount of memory needed in bytes
            const auto s = std::distance(begin, end);
            std::vector<size_t> offsets;
            offsets.reserve(s);
            offsets.push_back(0);
            for(auto it=begin; it!=end; ++it) {
                assert(*it >= 0);
                offsets.push_back( offsets.back() + *it );
            }
            return offsets;
        }

    template<typename T>
        void two_dim_variable_array<T>::clear()
        {
            data_.clear();
            offsets_.clear();
            offsets_.push_back(0);
        }

    template<typename T>
        template<typename ITERATOR>
        void two_dim_variable_array<T>::push_back(ITERATOR val_begin, ITERATOR val_end)
        {
            offsets_.push_back(offsets_.back() + std::distance(val_begin, val_end));
            for(auto it=val_begin; it!=val_end; ++it)
            {
                data_.push_back(*it);
            }
        }

} // namespace LPMP
