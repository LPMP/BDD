#pragma once

namespace LPMP {

template<class T>
struct tkahan
{
private:
   // value
   T val{ 0 };
   // pending correction
   T c{ 0 };

public:
   constexpr tkahan(T _val)
     : val(_val)
   {
   }

   // build with T value and T error
   constexpr tkahan(T _val, T _c)
     : val(_val)
     , c(_c)
   {
   }

   // empty
   constexpr tkahan()
   {
   }

   T value() const
   {
      return this->val;
   }

   T correction() const
   {
      return this->c;
   }

   explicit operator T() const
   {
      return val;
   }

   // copy assignment (for any valid element)
   template<class X>
   tkahan<T>& operator+=(const X& add)
   {
      // naive solution
      // this->val += add;  // will accumulate errors easily
      //
      // kahan operation
      T y = add - this->c;
      T t = this->val + y;
      this->c = (t - this->val) - y;
      this->val = t;
      // we must ensure that 'c' is never 'contaminated' by 'nan'
      // TODO: verify that this is REALLY safe... looks like.
      this->c = std::isnan(this->c) ? 0.0 : this->c;
      //
      return *this;
   }

   // reverse (unary minus)
   tkahan<T> operator-() const
   {
      return tkahan<T>(-this->val, -this->c);
   }

};

}
