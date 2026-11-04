// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apocharmm_c/CudaLangevinPistonIntegrator.h"
#include "apocharmm_c/detail/CudaLangevinPistonIntegratorHandle.h"
#include "apocharmm_c/detail/EnumConversion.h"
#include "apocharmm_c/detail/ErrorInternal.h"
#include "apocharmm_c/detail/SubscriberHandle.h"
#include "apocharmm_c/detail/Validation.h"

#include <memory>
#include <vector>

extern "C" apo_status apo_cuda_langevin_piston_integrator_create(
    apo_cuda_langevin_piston_integrator **out, const double time_step) {
  const char *function_name = "apo_cuda_langevin_piston_integrator_create";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::prepare_output_pointer<
                                    apo_cuda_langevin_piston_integrator>(
            out, function_name, "out"));

        std::unique_ptr<apo_cuda_langevin_piston_integrator> handle(
            new apo_cuda_langevin_piston_integrator());
        handle->object =
            std::make_shared<CudaLangevinPistonIntegrator>(time_step);
        handle->base.object = handle->object;

        *out = handle.release();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" void apo_cuda_langevin_piston_integrator_destroy(
    apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name = "apo_cuda_langevin_piston_integrator_destroy";
  apocharmm_c::guard_destroy(
      [integrator](void) -> void {
        delete integrator;
        return;
      },
      function_name);
  return;
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_use_nose_hoover_thermostat(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_use_nose_hoover_thermostat";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->useNoseHooverThermostat(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_reference_temperature(
    apo_cuda_langevin_piston_integrator *integrator, const double temperature) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_reference_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->setReferenceTemperature(temperature);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_nose_hoover_piston_mass(
    apo_cuda_langevin_piston_integrator *integrator, const double mass) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_nose_hoover_piston_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->setNoseHooverPistonMass(mass);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_langevin_piston_integrator_use_old_temperature(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_use_old_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->useOldTemperature(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_reference_pressure(
    apo_cuda_langevin_piston_integrator *integrator,
    const double *pressure_tensor, const size_t pressure_tensor_len) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_reference_pressure";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            pressure_tensor, function_name, "pressure_tensor"));

        if (pressure_tensor_len != 9) {
          return apocharmm_c::invalid_argument(
              function_name, "pressure_tensor must contain exactly 9 elements");
        }

        std::vector<double> reference_pressure(pressure_tensor,
                                               pressure_tensor + 9);

        integrator->object->setReferencePressure(reference_pressure);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_constant_surface_tension(
    apo_cuda_langevin_piston_integrator *integrator, const bool flag) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_constant_surface_tension";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->setConstantSurfaceTension(flag);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_mass(
    apo_cuda_langevin_piston_integrator *integrator, const double *mass,
    const size_t mass_len) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(mass, function_name, "mass"));

        if ((mass_len == 0) || (mass_len > 3)) {
          return apocharmm_c::invalid_argument(
              function_name, "mass must contain 1, 2, or 3 elements depending "
                             "on crystal type");
        }

        std::vector<double> piston_mass(mass, mass + mass_len);

        integrator->object->setLangevinPistonMass(piston_mass);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_friction_seed(
    apo_cuda_langevin_piston_integrator *integrator, const uint64_t seed) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_friction_seed";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->setLangevinPistonFrictionSeed(seed);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_set_langevin_piston_friction(
    apo_cuda_langevin_piston_integrator *integrator, const double friction) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_set_langevin_piston_friction";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->setLangevinPistonFriction(friction);

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_langevin_piston_integrator_reset_averages(
    apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_reset_averages";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        integrator->object->resetAverages();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_reference_temperature(
    double *temperature,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_reference_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        *temperature = integrator->object->getReferenceTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass(
    double *mass, const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_nose_hoover_piston_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<double>(mass, function_name, "mass"));

        CudaContainer<double> &piston_mass =
            integrator->object->getNoseHooverPistonMass();

        if (piston_mass.size() != 1) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR, function_name,
              "Nose-Hoover piston mass does not contain exactly 1 element");
        }

        piston_mass.transferToHost();
        *mass = piston_mass[0];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_average_temperature(
    double *temperature,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_average_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        CudaContainer<double> &average_temperature =
            integrator->object->getAverageTemperature();

        if (average_temperature.size() != 2) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR, function_name,
              "Average temperature does not contain exactly 2 elements");
        }

        average_temperature.transferToHost();

        if (integrator->object->usingOldTemperature())
          *temperature = average_temperature[0];
        else
          *temperature = average_temperature[1];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_reference_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_reference_pressure_tensor";
  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        constexpr size_t req_len = 9;

        CudaContainer<double> &reference_pressure_tensor =
            integrator->object->getReferencePressureTensor();

        if (reference_pressure_tensor.size() != req_len) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_cuda_langevin_piston_integrator_get_reference_pressure_"
              "tensor: reference pressure tensor does not contain 9 elements");
        }

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            pressure_tensor, pressure_tensor_len, req_len, function_name,
            "Reference pressure tensor buffer"));

        reference_pressure_tensor.transferToHost();

        for (size_t i = 0; i < req_len; i++)
          pressure_tensor[i] = reference_pressure_tensor[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_langevin_piston_integrator_get_crystal_type(
    apo_crystal_type *crystal_type,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_crystal_type";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::require_pointer<apo_crystal_type>(
                crystal_type, function_name, "crystal_type"));

        return apocharmm_c::from_crystal_type(
            crystal_type, integrator->object->getCrystalType(), function_name);
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_langevin_piston_mass(
    size_t *num_mass, double *mass, const size_t mass_len,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_langevin_piston_mass";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<size_t>(
            num_mass, function_name, "num_mass"));

        *num_mass = 0;

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        CudaContainer<double> &piston_mass =
            integrator->object->getLangevinPistonMass();

        const size_t req_len = piston_mass.size();

        if (req_len > 3) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_cuda_langevin_piston_integrator_get_langevin_piston_mass: "
              "Langevin piston mass contains more than 3 elements");
        }

        *num_mass = req_len;

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            mass, mass_len, req_len, function_name,
            "Langevin piston mass buffer"));

        if (req_len > 0)
          piston_mass.transferToHost();

        for (size_t i = 0; i < req_len; i++)
          mass[i] = piston_mass[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_tensor";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        constexpr size_t req_len = 9;

        CudaContainer<double> &instantaneous_pressure_tensor =
            integrator->object->getInstantaneousPressureTensor();

        if (instantaneous_pressure_tensor.size() != req_len) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_"
              "tensor: instantaneous pressure tensor does not contain 9 "
              "elements");
        }

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            pressure_tensor, pressure_tensor_len, req_len, function_name,
            "Instantaneous pressure tensor buffer"));

        instantaneous_pressure_tensor.transferToHost();

        for (size_t i = 0; i < req_len; i++)
          pressure_tensor[i] = instantaneous_pressure_tensor[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar(
    double *pressure_scalar,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_instantaneous_pressure_scalar";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            pressure_scalar, function_name, "pressure_scalar"));

        CudaContainer<double> &instantaneous_pressure_scalar =
            integrator->object->getInstantaneousPressureScalar();

        if (instantaneous_pressure_scalar.size() != 1) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR, function_name,
              "Instantaneous pressure scalar does not contain exactly 1 "
              "element");
        }

        instantaneous_pressure_scalar.transferToHost();
        *pressure_scalar = instantaneous_pressure_scalar[0];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_average_pressure_tensor(
    double *pressure_tensor, const size_t pressure_tensor_len,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_average_pressure_tensor";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        constexpr size_t req_len = 9;

        CudaContainer<double> &average_pressure_tensor =
            integrator->object->getAveragePressureTensor();

        if (average_pressure_tensor.size() != req_len) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR,
              "apo_cuda_langevin_piston_integrator_get_average_pressure_"
              "tensor: average pressure tensor does not contain 9 "
              "elements");
        }

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_output_buffer<double>(
            pressure_tensor, pressure_tensor_len, req_len, function_name,
            "Average pressure tensor buffer"));

        average_pressure_tensor.transferToHost();

        for (size_t i = 0; i < req_len; i++)
          pressure_tensor[i] = average_pressure_tensor[i];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_average_pressure_scalar(
    double *pressure_scalar,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_average_pressure_scalar";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            pressure_scalar, function_name, "pressure_scalar"));

        CudaContainer<double> &average_pressure_scalar =
            integrator->object->getAveragePressureScalar();

        if (average_pressure_scalar.size() != 1) {
          return apocharmm_c::set_last_error(
              APO_STATUS_RUNTIME_ERROR, function_name,
              "Average pressure scalar does not contain exactly 1 element");
        }

        average_pressure_scalar.transferToHost();
        *pressure_scalar = average_pressure_scalar[0];

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_langevin_piston_friction_seed(
    uint64_t *seed, const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_langevin_piston_friction_seed";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<uint64_t>(
            seed, function_name, "seed"));

        *seed = integrator->object->getLangevinPistonFrictionSeed();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status
apo_cuda_langevin_piston_integrator_get_instantaneous_temperature(
    double *temperature,
    const apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_get_instantaneous_temperature";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_pointer<double>(
            temperature, function_name, "temperature"));

        *temperature = integrator->object->getInstantaneousTemperature();

        return APO_STATUS_OK;
      },
      function_name);
}

extern "C" apo_status apo_cuda_langevin_piston_integrator_as_cuda_integrator(
    apo_cuda_integrator **out,
    apo_cuda_langevin_piston_integrator *integrator) {
  const char *function_name =
      "apo_cuda_langevin_piston_integrator_as_cuda_integrator";

  return apocharmm_c::guard(
      [&](void) -> apo_status {
        APOCHARMM_C_RETURN_IF_ERROR(
            apocharmm_c::prepare_output_pointer<apo_cuda_integrator>(
                out, function_name, "out"));

        APOCHARMM_C_RETURN_IF_ERROR(apocharmm_c::require_handle_object<
                                    apo_cuda_langevin_piston_integrator>(
            integrator, function_name, "CudaLangevinPistonIntegrator"));

        *out = &integrator->base;

        return APO_STATUS_OK;
      },
      function_name);
}
