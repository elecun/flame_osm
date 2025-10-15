/**
 * @file test.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-10-01
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_TEST_HPP_INCLUDED
#define FLAME_TEST_HPP_INCLUDED

#include <flame/component/object.hpp>

using namespace std;
using namespace flame::component;

class test : public flame::component::object {
    public:
        test() = default;
        virtual ~test() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message(const message_t& msg) override;

}; /* class */

EXPORT_COMPONENT_API


#endif