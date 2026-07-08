/**
 * @file osm.monolithic.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief OSM Monolithic Inference Component
 * @version 0.1
 * @date 2026-07-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef FLAME_OSM_MONOLITHIC_INFERENCE_HPP_INCLUDED
#define FLAME_OSM_MONOLITHIC_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>

using namespace std;
using namespace flame::component;

class osm_monolithic_inference : public flame::component::Object {
    public:
        osm_monolithic_inference();
        virtual ~osm_monolithic_inference() = default;

        /* default interface functions */
        bool onInit() override;
        void onLoop() override;
        void onClose() override;
        void onData(flame::component::ZData& data) override;
};

EXPORT_COMPONENT_API

#endif
