#pragma once

#include <iostream>
#include <fstream>

namespace LPMP
{
        struct joint_output
        {
                joint_output() {}

                void set_file_stream(const std::string &log_file)
                {
                        log_file_ = log_file;
                        file_stream_ = std::ofstream(log_file_, std::ios::trunc);
                }

                std::ofstream file_stream_;
                std::string log_file_;
                bool to_console_ = true;
        };

        template <typename T>
        joint_output &operator<<(joint_output &o, const T &var)
        {
                if (o.to_console_)
                        std::cout << var;
                if (!o.log_file_.empty())
                        o.file_stream_ << var;
                return o;
        }

        inline joint_output bdd_log;
}