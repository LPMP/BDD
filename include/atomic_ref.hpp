//------------------------------------------------------------------------------
// std::experimental::atomic_ref
//
// reference implementation for compilers which support GNU atomic builtins
// https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
//
//------------------------------------------------------------------------------

#ifndef ATOMIC_REF_HPP
#define ATOMIC_REF_HPP

#include <atomic>
#include <type_traits>
#include <cstdint>
#include <cmath>

#if defined( _MSC_VER ) //msvc
  #error "Error: MSVC not currently supported"
#endif

#ifndef ATOMIC_REF_FORCEINLINE
  #define ATOMIC_REF_FORCEINLINE inline __attribute__((always_inline))
#endif

static_assert(  (__ATOMIC_RELAXED == std::memory_order_relaxed )
             && (__ATOMIC_CONSUME == std::memory_order_consume )
             && (__ATOMIC_ACQUIRE == std::memory_order_acquire )
             && (__ATOMIC_RELEASE == std::memory_order_release )
             && (__ATOMIC_ACQ_REL == std::memory_order_acq_rel )
             && (__ATOMIC_SEQ_CST == std::memory_order_seq_cst )
             , "Error: std::memory_order values are not equivalent to builtins"
             );

namespace Foo {

namespace Impl {

//------------------------------------------------------------------------------
template <typename T>
inline constexpr size_t atomic_ref_required_alignment_v = sizeof(T) == sizeof(uint8_t)  ? sizeof(uint8_t)
                                                        : sizeof(T) == sizeof(uint16_t) ? sizeof(uint16_t)
                                                        : sizeof(T) == sizeof(uint32_t) ? sizeof(uint32_t)
                                                        : sizeof(T) == sizeof(uint64_t) ? sizeof(uint64_t)
                                                        : sizeof(T) == sizeof(__uint128_t) ? sizeof(__uint128_t)
                                                        : std::alignment_of_v<T>
                                                        ;

template <typename T>
inline constexpr bool atomic_use_native_ops_v =  sizeof(T) <= sizeof(__uint128_t)
                                              && (  std::is_integral_v<T>
                                                 || std::is_enum_v<T>
                                                 || std::is_pointer_v<T>
                                                 )
                                              ;

template <typename T>
inline constexpr bool atomic_use_cast_ops_v =  !atomic_use_native_ops_v<T>
                                            && (  sizeof(T) == sizeof(uint8_t)
                                               || sizeof(T) == sizeof(uint16_t)
                                               || sizeof(T) == sizeof(uint32_t)
                                               || sizeof(T) == sizeof(uint64_t)
                                               || sizeof(T) == sizeof(__uint128_t)
                                               )
                                            ;

template <typename T>
using atomic_ref_cast_t = std::conditional_t< sizeof(T) == sizeof(uint8_t),  uint8_t
                        , std::conditional_t< sizeof(T) == sizeof(uint16_t), uint16_t
                        , std::conditional_t< sizeof(T) == sizeof(uint32_t), uint32_t
                        , std::conditional_t< sizeof(T) == sizeof(uint64_t), uint64_t
                        , std::conditional_t< sizeof(T) == sizeof(__uint128_t), __uint128_t
                        , T
                        >>>>>
                        ;

//------------------------------------------------------------------------------
// atomic_ref_ops: generic
//------------------------------------------------------------------------------
template <typename Base, typename ValueType, typename Enable = void>
struct atomic_ref_ops
{};


//------------------------------------------------------------------------------
// atomic_ref_ops: integral
//------------------------------------------------------------------------------
template <typename Base, typename ValueType>
struct atomic_ref_ops< Base, ValueType
                     , std::enable_if_t<  std::is_integral_v<ValueType>
                                       && !std::is_same_v<bool,ValueType>
                                       >
                     >
{
 public:
   using difference_type = ValueType;

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_add( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    return __atomic_fetch_add( static_cast<const Base*>(this)->ptr_
                             , val
                             , order
                             );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_sub( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    return __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_
                             , val
                             , order
                             );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_and( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    return __atomic_fetch_and( static_cast<const Base*>(this)->ptr_
                             , val
                             , order
                             );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_or( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    return __atomic_fetch_or( static_cast<const Base*>(this)->ptr_
                            , val
                            , order
                            );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_xor( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    return __atomic_fetch_xor( static_cast<const Base*>(this)->ptr_
                             , val
                             , order
                             );
  }


  ATOMIC_REF_FORCEINLINE
  difference_type operator++(int) const noexcept
  {
    return __atomic_fetch_add( static_cast<const Base*>(this)->ptr_, static_cast<difference_type>(1), std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator--(int) const noexcept
  {
    return __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_, static_cast<difference_type>(1), std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator++() const noexcept
  {
    return __atomic_add_fetch( static_cast<const Base*>(this)->ptr_, static_cast<difference_type>(1), std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator--() const noexcept
  {
    return __atomic_sub_fetch( static_cast<const Base*>(this)->ptr_, static_cast<difference_type>(1), std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator+=(difference_type val) const noexcept
  {
    return __atomic_add_fetch( static_cast<const Base*>(this)->ptr_, val, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator-=(difference_type val) const noexcept
  {
    return __atomic_sub_fetch( static_cast<const Base*>(this)->ptr_, val, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator&=(difference_type val) const noexcept
  {
    return __atomic_sub_fetch( static_cast<const Base*>(this)->ptr_, val, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator|=(difference_type val) const noexcept
  {
    return __atomic_or_fetch( static_cast<const Base*>(this)->ptr_, val, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator^=(difference_type val) const noexcept
  {
    return __atomic_xor_fetch( static_cast<const Base*>(this)->ptr_, val, std::memory_order_seq_cst );
  }
};


//------------------------------------------------------------------------------
// atomic_ref_ops: floating-point
//------------------------------------------------------------------------------
template <typename Base, typename ValueType>
struct atomic_ref_ops< Base, ValueType
                     , std::enable_if_t<  std::is_floating_point_v<ValueType> >
                     >
{
 public:
   using difference_type = ValueType;

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_add( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    difference_type expected = static_cast<const Base*>(this)->load(std::memory_order_relaxed);
    difference_type desired = expected + val;

    while (! static_cast<const Base*>(this)->
              compare_exchange_weak( expected
                                   , desired
                                   , order
                                   , std::memory_order_relaxed
                                   )
         )
    {
      desired = expected + val;
      if (std::isnan(expected)) break;
    }

    return expected;
  }

  ATOMIC_REF_FORCEINLINE
  difference_type fetch_sub( difference_type val
                           , std::memory_order order = std::memory_order_seq_cst
                           ) const noexcept
  {
    difference_type expected = static_cast<const Base*>(this)->load(std::memory_order_relaxed);
    difference_type desired = expected - val;

    while (! static_cast<const Base*>(this)->
              compare_exchange_weak( expected
                                   , desired
                                   , order
                                   , std::memory_order_relaxed
                                   )
         )
    {
      desired = expected - val;
      if (std::isnan(expected)) break;
    }

    return expected;
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator+=(difference_type val) const noexcept
  {
    return fetch_add( val ) + val;
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator-=(difference_type val) const noexcept
  {
    return fetch_sub( val ) - val;
  }
};


//------------------------------------------------------------------------------
// atomic_ref_ops: pointer to object
//------------------------------------------------------------------------------
template <typename Base, typename ValueType>
struct atomic_ref_ops< Base, ValueType
                     , std::enable_if<  std::is_pointer_v<ValueType>
                                     && std::is_object_v< std::remove_pointer_t<ValueType>>
                                     >
                     >
{
  static constexpr ptrdiff_t stride = static_cast<ptrdiff_t>(sizeof( std::remove_pointer_t<ValueType> ));

 public:
  using difference_type = ptrdiff_t;

  ATOMIC_REF_FORCEINLINE
  ValueType fetch_add( difference_type val
                     , std::memory_order order = std::memory_order_seq_cst
                     ) const noexcept
  {
    return val >= 0
           ? __atomic_fetch_add( static_cast<const Base*>(this)->ptr_
                                , stride*val
                                , order
                               )
           : __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_
                               , -(stride*val)
                               , order
                               )
           ;
  }

  ATOMIC_REF_FORCEINLINE
  ValueType fetch_sub( difference_type val
                     , std::memory_order order = std::memory_order_seq_cst
                     ) const noexcept
  {
    return val >= 0
           ? __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_
                                , stride*val
                                , order
                               )
           : __atomic_fetch_add( static_cast<const Base*>(this)->ptr_
                               , -(stride*val)
                               , order
                               )
           ;
  }



  ATOMIC_REF_FORCEINLINE
  difference_type operator++(int) const noexcept
  {
    return __atomic_fetch_add( static_cast<const Base*>(this)->ptr_, stride, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator--(int) const noexcept
  {
    return __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_, stride, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator++() const noexcept
  {
    return __atomic_add_fetch( static_cast<const Base*>(this)->ptr_, stride, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator--() const noexcept
  {
    return __atomic_sub_fetch( static_cast<const Base*>(this)->ptr_, stride, std::memory_order_seq_cst );
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator+=(difference_type val) const noexcept
  {
    return val >= 0
           ? __atomic_fetch_add( static_cast<const Base*>(this)->ptr_
                                , stride*val
                                , std::memory_order_seq_cst
                               )
           : __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_
                               , -(stride*val)
                                , std::memory_order_seq_cst
                               )
           ;
  }

  ATOMIC_REF_FORCEINLINE
  difference_type operator-=(difference_type val) const noexcept
  {
    return val >= 0
           ? __atomic_fetch_sub( static_cast<const Base*>(this)->ptr_
                                , stride*val
                                , std::memory_order_seq_cst
                               )
           : __atomic_fetch_add( static_cast<const Base*>(this)->ptr_
                               , -(stride*val)
                                , std::memory_order_seq_cst
                               )
           ;
  }
};

} // namespace Impl

template < class T >
struct atomic_ref
  : public Impl::atomic_ref_ops< atomic_ref<T>, T >
{
  static_assert( std::is_trivially_copyable_v<T>
               , "Error: atomic_ref<T> requires T to be trivially copyable");

private:
  T* ptr_;

  friend struct Impl::atomic_ref_ops< atomic_ref<T>, T>;

public:

  using value_type = T;

  static constexpr size_t required_alignment  = Impl::atomic_ref_required_alignment_v<T>;
  static constexpr bool   is_always_lock_free = __atomic_always_lock_free( sizeof(T) <= required_alignment
                                                                                      ? required_alignment 
                                                                                      : sizeof(T)
                                                                         , nullptr
                                                                         );

  atomic_ref() = delete;
  atomic_ref & operator=( const atomic_ref & ) = delete;

  ATOMIC_REF_FORCEINLINE
  explicit atomic_ref( value_type & obj )
    : ptr_{&obj}
  {}

  ATOMIC_REF_FORCEINLINE
  atomic_ref( const atomic_ref & ref ) noexcept = default;

  ATOMIC_REF_FORCEINLINE
  value_type operator=( value_type desired ) const noexcept
  {
    store(desired);
    return desired;
  }

  ATOMIC_REF_FORCEINLINE
  operator value_type() const noexcept
  {
    return load();
  }

  ATOMIC_REF_FORCEINLINE
  bool is_lock_free() const noexcept
  {
    return __atomic_is_lock_free( sizeof(value_type), ptr_ );
  }

  ATOMIC_REF_FORCEINLINE
  void store( value_type desired
            , std::memory_order order = std::memory_order_seq_cst
            ) const noexcept
  {
    if constexpr ( Impl::atomic_use_native_ops_v<T> ) {
      __atomic_store_n( ptr_, desired, order );
    }
    else if constexpr ( Impl::atomic_use_cast_ops_v<T> ) {
      typedef Impl::atomic_ref_cast_t<T> __attribute__((__may_alias__)) cast_type;

      __atomic_store_n( reinterpret_cast<cast_type*>(ptr_)
                      , *reinterpret_cast<cast_type*>(&desired)
                      , order
                      );
    }
    else {
      __atomic_store( ptr_, &desired, order );
    }
  }

  ATOMIC_REF_FORCEINLINE
  value_type load( std::memory_order order = std::memory_order_seq_cst ) const noexcept
  {
    if constexpr ( Impl::atomic_use_native_ops_v<T> ) {
      return __atomic_load_n( ptr_, order );
    }
    else if constexpr ( Impl::atomic_use_cast_ops_v<T> ) {
      typedef Impl::atomic_ref_cast_t<T> __attribute__((__may_alias__)) cast_type;

      cast_type tmp = __atomic_load_n( reinterpret_cast<cast_type*>(ptr_)
                                     , order
                                     );
      return *reinterpret_cast<value_type*>(&tmp);
    }
    else {
      value_type result;
      __atomic_load( ptr_, &result, order );
      return result;
    }
  }

  ATOMIC_REF_FORCEINLINE
  value_type exchange( value_type desired
                     , std::memory_order order = std::memory_order_seq_cst
                     ) const noexcept
  {
    if constexpr ( Impl::atomic_use_native_ops_v<T> ) {
      return __atomic_exchange_n( ptr_, desired, order );
    }
    else if constexpr ( Impl::atomic_use_cast_ops_v<T> ) {
      typedef Impl::atomic_ref_cast_t<T> __attribute__((__may_alias__)) cast_type;

      cast_type tmp = __atomic_exchange_n( reinterpret_cast<cast_type*>(ptr_)
                                         , *reinterpret_cast<cast_type*>(&desired)
                                         , order
                                         );
      return *reinterpret_cast<value_type*>(&tmp);
    }
    else {
      value_type result;
      __atomic_exchange( ptr_, &desired, &result, order);
      return result;
    }
  }

  ATOMIC_REF_FORCEINLINE
  bool compare_exchange_weak( value_type& expected
                            , value_type desired
                            , std::memory_order success
                            , std::memory_order failure
                            ) const noexcept
  {
    if constexpr ( Impl::atomic_use_native_ops_v<T> ) {
      return __atomic_compare_exchange_n( ptr_, &expected, desired, true, success, failure );
    }
    else if constexpr ( Impl::atomic_use_cast_ops_v<T> ) {
      typedef Impl::atomic_ref_cast_t<T> __attribute__((__may_alias__)) cast_type;

      return __atomic_compare_exchange_n( reinterpret_cast<cast_type*>(ptr_)
                                        , reinterpret_cast<cast_type*>(&expected)
                                        , *reinterpret_cast<cast_type*>(&desired)
                                        , true
                                        , success
                                        , failure
                                        );
    }
    else {
      return __atomic_compare_exchange( ptr_, &expected, &desired, true, success, failure );
    }
  }

  ATOMIC_REF_FORCEINLINE
  bool compare_exchange_strong( value_type& expected
                              , value_type desired
                              , std::memory_order success
                              , std::memory_order failure
                              ) const noexcept
  {
    if constexpr ( Impl::atomic_use_native_ops_v<T> ) {
      return __atomic_compare_exchange_n( ptr_, &expected, desired, false, success, failure );
    }
    else if constexpr ( Impl::atomic_use_cast_ops_v<T> ) {
      typedef Impl::atomic_ref_cast_t<T> __attribute__((__may_alias__)) cast_type;

      return __atomic_compare_exchange_n( reinterpret_cast<cast_type*>(ptr_)
                                        , reinterpret_cast<cast_type*>(&expected)
                                        , *reinterpret_cast<cast_type*>(&desired)
                                        , false
                                        , success
                                        , failure
                                        );
    }
    else {
      return __atomic_compare_exchange( ptr_, &expected, &desired, false, success, failure );
    }
  }

  ATOMIC_REF_FORCEINLINE
  bool compare_exchange_weak( value_type& expected
                            , value_type desired
                            , std::memory_order order = std::memory_order_seq_cst
                            ) const noexcept
  {
    return compare_exchange_weak( expected, desired, order, order );
  }

  ATOMIC_REF_FORCEINLINE
  bool compare_exchange_strong( value_type& expected
                              , value_type desired
                              , std::memory_order order = std::memory_order_seq_cst
                              ) const noexcept
  {
    return compare_exchange_strong( expected, desired, order, order );
  }
};

} // namespace Foo


#endif // ATOMIC_REF_HPP
