//
//  SNN.cpp
//  SNN
//
//  Created by 전민기 on 2022/12/26.
//

#include "SNN.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <time.h>
#include <random>
#include <thread>
#include <arm_neon.h>
#include <stdlib.h>
#include <unordered_map>
#include <math.h>
#include "/opt/homebrew/Cellar/llvm/15.0.6/lib/clang/15.0.6/include/omp.h"

void SNN::set_initial(int iteration){
    this->iteration = iteration;
}

/*void SNN::setting_for_proceeding(int time){
    if (while_iter == 0) {
#pragma omp parallel
        {
#pragma omp for
            {
                for (int i = 0; i<nE_; i+=4) {
                    float32x4_t temp_E_potential = vld1q_f32(&E_potential_->at(i));
                    float32x4_t temp_I_potential = vld1q_f32(&I_potential_->at(i));
                    float32x4_t temp_E_Con_E = vld1q_f32(&E_Con_E_->at(i));
                    float32x4_t temp_E_Con_I = vld1q_f32(&E_Con_I_->at(i));
                    float32x4_t temp_I_Con_E = vld1q_f32(&I_Con_E_->at(i));
                    float32x4_t temp_E_dCon_E = vld1q_f32(&E_dCon_E_->at(i));
                    vst1q_f32(&E_potential_mem->at(i), temp_E_potential);
                    vst1q_f32(&I_potential_mem->at(i), temp_I_potential);
                    vst1q_f32(&E_Con_E_mem->at(i), temp_E_Con_E);
                    vst1q_f32(&E_Con_I_mem->at(i), temp_E_Con_I);
                    vst1q_f32(&I_Con_E_mem->at(i), temp_I_Con_E);
                    vst1q_f32(&E_dCon_E_mem->at(i), temp_E_dCon_E);
                }
            }
        }
        while_iter += 1;
    }
    else{
#pragma omp parallel
        {
            float32x4_t zero4 = vdupq_n_f32(0);
#pragma omp for
            {
                for (int i = 0; i < nE_; i+=4) {
                    vst1q_f32(&(E_spike_total_->at(i)), zero4);
                }
            }
#pragma omp for
            {
                for (int i = 0; i < nE_*time; i+=4) {
                    vst1q_f32(&(E_spike_->at(i)), zero4);
                    vst1q_f32(&(input_data_->at(i)), zero4);
                }
            }
#pragma omp for
            {
                for (int i = 0; i < nE_*(time+1); i+=4) {
                    vst1q_f32(&(I_spike_->at(i)), zero4);
                }
            }
#pragma omp for
            {
                for (int i = 0; i<nE_; i+=4) {
                    float32x4_t temp_E_potential = vld1q_f32(&E_potential_mem->at(i));
                    float32x4_t temp_I_potential = vld1q_f32(&I_potential_mem->at(i));
                    float32x4_t temp_E_Con_E = vld1q_f32(&E_Con_E_mem->at(i));
                    float32x4_t temp_E_Con_I = vld1q_f32(&E_Con_I_mem->at(i));
                    float32x4_t temp_I_Con_E = vld1q_f32(&I_Con_E_mem->at(i));
                    float32x4_t temp_E_dCon_E = vld1q_f32(&E_dCon_E_mem->at(i));
                    vst1q_f32(&E_potential_->at(i), temp_E_potential);
                    vst1q_f32(&I_potential_->at(i), temp_I_potential);
                    vst1q_f32(&E_Con_E_->at(i), temp_E_Con_E);
                    vst1q_f32(&E_Con_I_->at(i), temp_E_Con_I);
                    vst1q_f32(&I_Con_E_->at(i), temp_I_Con_E);
                    vst1q_f32(&E_dCon_E_->at(i), temp_E_dCon_E);
                }
            }
        }
    }
}*/

void SNN::poisson_spike_generator(int time){
    int nE_ = this->nE_;
    int nInput_ = this->nInput_;
    int interation_ = this -> iteration;
    float interval_ = this->interval_;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    float a = 0.000125;
#pragma omp parallel
    {
        float32x4_t scalar_vector = vdupq_n_f32(a*interval_);
#pragma omp for collapse(2)
        {
            for (int j = 0; j < time; ++j){
                for (int i=nInput_*interation_; i<nInput_*(interation_+1); i+=4)  {
                    float32x4_t v1 = vld1q_f32(&train_data_.at(i));
                    int input_neuron = i % (nInput_*interation_);
                    float32x4_t rand_vec = vsetq_lane_f32(dis(gen), vdupq_n_f32(0), 0);
                    rand_vec = vsetq_lane_f32(dis(gen), rand_vec, 1);
                    rand_vec = vsetq_lane_f32(dis(gen), rand_vec, 2);
                    rand_vec = vsetq_lane_f32(dis(gen), rand_vec, 3);
                    float32x4_t v2 = vmulq_f32(v1, scalar_vector);
                    uint32x4_t result1 = vcgtq_f32(v2, rand_vec);
                    uint32x4_t result = vshrq_n_u32(result1, 31);
                    float32x4_t result_vector = vcvtq_f32_u32(result);
                    vst1q_f32(&train_data_temp->at(j*nInput_+input_neuron), result_vector);
                }
            }
        }
#pragma omp for collapse(2)
        {
            for (int k = 0; k<time; ++k) {
                for (int i = 0; i < nInput_; i+=4) {
                    float32x4_t train4 = vld1q_f32(&train_data_temp->at(k*nInput_+i));
                    float32x4_t weight;
                    if (vaddvq_f32(train4) != 0.0f) {
                        for (int j = 0; j < nE_; ++j)  {
                            weight[0]=  In_E_weight_->at(i*nE_ + j);
                            weight[1]=  In_E_weight_->at((i+1)*nE_ + j);
                            weight[2]=  In_E_weight_->at((i+2)*nE_ + j);
                            weight[3]=  In_E_weight_->at((i+3)*nE_ + j);
                            float32x4_t temp_input4 = vmulq_f32(weight, train4);
                            input_data_->at(k*nE_+j) += vaddvq_f32(temp_input4);
                        }
                    }
                }
            }
        }
    }
}

void SNN::initializatoin(float &E_total_spike, int time, int iteration){
    int nE_ = this->nE_;
#pragma omp parallel
    {
        float32x4_t zero4 = vdupq_n_f32(0);
#pragma omp for
        {
            for (int i = 0; i < nE_; i+=4) {
                vst1q_f32(&(E_spike_total_->at(i)), zero4);
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_*time; i+=4) {
                vst1q_f32(&(E_spike_previous->at(i)), vld1q_f32(&(E_spike_->at(i))));
                vst1q_f32(&(E_spike_->at(i)), zero4);
                vst1q_f32(&(input_data_->at(i)), zero4);
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_*(time+1); i+=4) {
                vst1q_f32(&(I_spike_previous->at(i)), vld1q_f32(&(I_spike_->at(i))));
                vst1q_f32(&(I_spike_->at(i)), zero4);
            }
        }
    }
    E_total_spike = 0;
}

void SNN::Stimulation(int time, vector<float> *after_save){
    float gE_constant = (1 - time_step / tau_syn_E);
    float gI_constant = (1 - time_step / tau_syn_I);
    float E_time_con = time_step/tau_E;
    float I_time_con = time_step/tau_I;
    int nE_ = this->nE_;
    int nI_ = this->nI_;
    vector<float> I_data_sum(nE_, 0);
    vector<float> total(nE_,0);
    vector<float> temp(nE_,0);
    float temp_spike = 0;
    float32x4_t temp4 = vdupq_n_f32(0.0f);
    float32x4_t gL4 = vdupq_n_f32(gL);
    float32x4_t v_thresh_E4 = vdupq_n_f32(v_thresh_E);
    float32x4_t v_thresh_I4 = vdupq_n_f32(v_thresh_I);
    float32x4_t v_reset_E4 = vdupq_n_f32(v_reset_E);
    float32x4_t v_reset_I4 = vdupq_n_f32(v_reset_I);
    float32x4_t v_rest_E4 = vdupq_n_f32(v_rest_E);
    float32x4_t v_rest_I4 = vdupq_n_f32(v_rest_I);
#pragma omp parallel
    {
        float32x4_t verifying_E = temp4;
        float32x4_t verifying_I = temp4;
        for (int i = 0; i < time; ++i) {
#pragma omp single nowait
            {
                temp_spike = 0;
            }
#pragma omp for
            {
                for (int j = 0; j < nE_; j+=4){
                    float32x4_t E_rest = vdupq_n_f32(v_rest_E);
                    float32x4_t E_vE_E = vdupq_n_f32(vE_E);
                    float32x4_t E_vI_E = vdupq_n_f32(vI_E);
                    
                    if (i < 5) {
                        float32x4_t E_Potential4 = vld1q_f32(&(E_potential_->at(j)));
                        if (i == 0) {
                            verifying_E = {vaddvq_f32(vld1q_f32(&E_spike_previous->at(j*time + time - 5))) + E_spike_previous->at(j*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+1)*time + time - 5))) + E_spike_previous->at((j+1)*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+2)*time + time - 5))) + E_spike_previous->at((j+2)*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+3)*time + time - 5))) + E_spike_previous->at((j+3)*time + time - 1)};
                        }
                        else {
                            for (int k = 5; k > 0; --k) {
                                verifying_E[0] = verifying_E[0] - E_spike_previous->at(j*time + time - k) + E_spike_-> at(j*time + time - 5 - k);
                                verifying_E[1] = verifying_E[1] - E_spike_previous->at((j+1)*time + time - k) + E_spike_-> at((j+1)*time + time - 5 - k);
                                verifying_E[2] = verifying_E[2] - E_spike_previous->at((j+2)*time + time - k) + E_spike_-> at((j+2)*time + time - 5 - k);
                                verifying_E[3] = verifying_E[3] - E_spike_previous->at((j+3)*time + time - k) + E_spike_-> at((j+3)*time + time - 5 - k);
                            }
                        }
                        
                        float32x4_t E_Con_E4 = vld1q_f32(&(E_Con_E_->at(j)));
                        float32x4_t input_data4;
                        float32x4_t E_dCon_E4;
                        float32x4_t Input_E_spike_data;
                        input_data4 = vld1q_f32(&(input_data_->at(i*nE_+j)));
                        E_Con_E4 = vmulq_n_f32(E_Con_E4, gE_constant);
                        E_dCon_E4 = vld1q_f32(&(E_dCon_E_->at(j)));
                        Input_E_spike_data = vmulq_f32(E_dCon_E4, input_data4);
                        E_Con_E4 = vaddq_f32(E_Con_E4, Input_E_spike_data);
                        vst1q_f32(&(E_Con_E_->at(j)), E_Con_E4);

                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /* Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                        float32x4_t I_E_spike_data;
                        for (int k = 0; k < nI_; ++k) {
                            if (I_spike_->at(k*time+i) == 1) {
                                float32x4_t weight_I_E4 = {I_E_weight_->at(k*nE_+j), I_E_weight_->at(k*nE_+j+1), I_E_weight_->at(k*nE_+j+2), I_E_weight_->at(k*nE_+j+3)};
                                float32x4_t I_spike4 = vdupq_n_f32(I_spike_->at(k*time+i));
                                I_E_spike_data = vmulq_f32(weight_I_E4, I_spike4);
                                I_data_sum.at(j) += I_E_spike_data[0];
                                I_data_sum.at(j+1) += I_E_spike_data[1];
                                I_data_sum.at(j+2) += I_E_spike_data[2];
                                I_data_sum.at(j+3) += I_E_spike_data[3];
                            }
                        }
                        float32x4_t E_dCon_I4 = vld1q_f32(&(E_dCon_I_->at(j)));
                        I_E_spike_data = vld1q_f32(&(I_data_sum.at(j)));
                        I_E_spike_data = vmulq_f32(I_E_spike_data, E_dCon_I4);
                        
                        float32x4_t E_Con_I4 = vld1q_f32(&(E_Con_I_->at(j)));
                        E_Con_I4 = vmulq_n_f32(E_Con_I4, gI_constant);
                        
                        E_Con_I4 = vaddq_f32(E_Con_I4, I_E_spike_data);
                        vst1q_f32(&(E_Con_I_->at(j)), E_Con_I4);
                        vst1q_f32(&(I_data_sum.at(j)), temp4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step /tau_e) */
                        
                        E_rest = vsubq_f32(E_rest, E_Potential4);
                        
                        E_vE_E = vsubq_f32(E_vE_E, E_Potential4);
                        E_Con_E4 = vdivq_f32(E_Con_E4, gL4);
                        
                        E_vI_E = vsubq_f32(E_vI_E, E_Potential4);
                        
                        E_Con_I4 = vdivq_f32(E_Con_I4, gL4);
                        
                        float32x4_t E_dPotential = vmlaq_f32(E_rest, E_Con_E4, E_vE_E);
                        E_dPotential = vmlaq_f32(E_dPotential, E_Con_I4, E_vI_E);
                        
                        E_dPotential = vmulq_n_f32(E_dPotential, E_time_con);
                        E_Potential4 = vaddq_f32(E_Potential4, E_dPotential);
                        E_Potential4 = vmaxq_f32(v_reset_E4, E_Potential4);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_E[k] != 0 and E_Potential4[k] > v_rest_E) {
                                E_Potential4[k] = v_rest_E;
                            }
                        }
                        uint32x4_t E_spike4_1 = vcgtq_f32(E_Potential4, v_thresh_E4);
                        uint32x4_t E_spike4_int = vshrq_n_u32(E_spike4_1, 31);
                        float32x4_t E_spike4 = vcvtq_f32_u32(E_spike4_int);

                        E_spike_->at(j*350 + i) = E_spike4[0];
                        E_spike_->at((j+1)*350 + i) = E_spike4[1];
                        E_spike_->at((j+2)*350 + i) = E_spike4[2];
                        E_spike_->at((j+3)*350 + i) = E_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (E_spike4[k] == 1) {
                                E_Potential4[k] = v_reset_E;
                            }
                        }
                        vst1q_f32(&(E_potential_->at(j)), E_Potential4);
                        /* END */
                    }
                    else{
                        float32x4_t verifying_0 = vld1q_f32(&(E_spike_->at(j*350 + i-5)));
                        float32x4_t verifying_1 = vld1q_f32(&(E_spike_->at((j+1)*350 + i-5)));
                        float32x4_t verifying_2 = vld1q_f32(&(E_spike_->at((j+2)*350 + i-5)));
                        float32x4_t verifying_3 = vld1q_f32(&(E_spike_->at((j+3)*350 + i-5)));
                        verifying_E = {vaddvq_f32(verifying_0) + E_spike_->at(j*350 + i-1), vaddvq_f32(verifying_1) + E_spike_->at((j+1)*350 + i-1), vaddvq_f32(verifying_2) + E_spike_->at((j+2)*350 + i-1), vaddvq_f32(verifying_3) + E_spike_->at((j+3)*350 + i-1)};
                        float32x4_t E_Potential4 = vld1q_f32(&(E_potential_->at(j)));
                        
                        /* Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max */
                        float32x4_t E_Con_E4 = vld1q_f32(&(E_Con_E_->at(j)));
                        float32x4_t input_data4;
                        float32x4_t E_dCon_E4;
                        float32x4_t Input_E_spike_data;
                        input_data4 = vld1q_f32(&(input_data_->at(i*nE_+j)));
                        E_Con_E4 = vmulq_n_f32(E_Con_E4, gE_constant);
                        E_dCon_E4 = vld1q_f32(&(E_dCon_E_->at(j)));
                        Input_E_spike_data = vmulq_f32(E_dCon_E4, input_data4);
                        E_Con_E4 = vaddq_f32(E_Con_E4, Input_E_spike_data);
                        vst1q_f32(&(E_Con_E_->at(j)), E_Con_E4);
                        
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /* Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                        float32x4_t I_E_spike_data;
                        for (int k = 0; k < nI_; ++k) {
                            if (I_spike_->at(k*time+i) == 1) {
                                float32x4_t weight_I_E4 = {I_E_weight_->at(k*nE_+j), I_E_weight_->at(k*nE_+j+1), I_E_weight_->at(k*nE_+j+2), I_E_weight_->at(k*nE_+j+3)};
                                float32x4_t I_spike4 = vdupq_n_f32(I_spike_->at(k*time+i));
                                I_E_spike_data = vmulq_f32(weight_I_E4, I_spike4);
                                I_data_sum.at(j) += I_E_spike_data[0];
                                I_data_sum.at(j+1) += I_E_spike_data[1];
                                I_data_sum.at(j+2) += I_E_spike_data[2];
                                I_data_sum.at(j+3) += I_E_spike_data[3];
                            }
                        }
                        float32x4_t E_dCon_I4 = vld1q_f32(&(E_dCon_I_->at(j)));
                        I_E_spike_data = vld1q_f32(&(I_data_sum.at(j)));
                        I_E_spike_data = vmulq_f32(I_E_spike_data, E_dCon_I4);
                        
                        float32x4_t E_Con_I4 = vld1q_f32(&(E_Con_I_->at(j)));
                        E_Con_I4 = vmulq_n_f32(E_Con_I4, gI_constant);
                        
                        E_Con_I4 = vaddq_f32(E_Con_I4, I_E_spike_data);
                        vst1q_f32(&(E_Con_I_->at(j)), E_Con_I4);
                        vst1q_f32(&(I_data_sum.at(j)), temp4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step /tau_e) */
                        E_rest = vsubq_f32(E_rest, E_Potential4);
                        
                        E_vE_E = vsubq_f32(E_vE_E, E_Potential4);
                        E_Con_E4 = vdivq_f32(E_Con_E4, gL4);
                        
                        E_vI_E = vsubq_f32(E_vI_E, E_Potential4);
                        E_Con_I4 = vdivq_f32(E_Con_I4, gL4);
                    
                        float32x4_t E_dPotential = vmlaq_f32(E_rest, E_Con_E4, E_vE_E);
                        E_dPotential = vmlaq_f32(E_dPotential, E_Con_I4, E_vI_E);
                        
                        E_dPotential = vmulq_n_f32(E_dPotential, E_time_con);
                        E_Potential4 = vaddq_f32(E_Potential4, E_dPotential);
                        E_Potential4 = vmaxq_f32(v_reset_E4, E_Potential4);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_E[k] != 0 and E_Potential4[k] > v_rest_E) {
                                E_Potential4[k] = v_rest_E;
                            }
                        }
                        uint32x4_t E_spike4_1 = vcgtq_f32(E_Potential4, v_thresh_E4);
                        uint32x4_t E_spike4_int = vshrq_n_u32(E_spike4_1, 31);
                        float32x4_t E_spike4 = vcvtq_f32_u32(E_spike4_int);
                        E_spike_->at(j*350 + i) = E_spike4[0];
                        E_spike_->at((j+1)*350 + i) = E_spike4[1];
                        E_spike_->at((j+2)*350 + i) = E_spike4[2];
                        E_spike_->at((j+3)*350 + i) = E_spike4[3];
                        
                        for (int k = 0; k < 4; ++k) {
                            if (E_spike4[k] == 1) {
                                E_Potential4[k] = v_reset_E;
                            }
                        }
                        vst1q_f32(&(E_potential_->at(j)), E_Potential4);
                        /* end */
                    }
                }
            }
            float temp_spike_private = 0;
#pragma omp for
            {
                for (int j = 0; j < nE_; ++j){
                    if (E_spike_->at(j*350 + i) == 1) {
                        temp_spike_private += 1;
                        break;
                    }
                }
            }
#pragma omp critical
            {
                temp_spike += temp_spike_private;
            }
#pragma omp for
            {
                for (int j = 0; j < nE_; ++j){
                    if (temp_spike != 0) {
                        if (E_spike_->at(j*350 + i) == 1) {
                            if (rate_->at(j) != 1) {
                                E_dCon_E_->at(j) *= rate_->at(j);
                                if (E_dCon_E_->at(j) > 1) {
                                    E_dCon_E_->at(j) = 1;
                                }
                            }
                        }
                        else {
                            if (after_save->at(j) != 0.0f) {
                                float E_dcon_E_temp = E_dCon_E_->at(j);
                                //loat E_dcon_E_temp_temp = E_dcon_E_temp*rate_temp;
                                float E_dcon_E_temp_temp_tmep = 1-E_dcon_E_temp;
                                E_dcon_E_temp_temp_tmep = after_save->at(j)*E_dcon_E_temp_temp_tmep;
                                E_dcon_E_temp = E_dcon_E_temp+ E_dcon_E_temp_temp_tmep;
                                E_dCon_E_->at(j) =  E_dcon_E_temp;
                            }
                        }
                    }
                }
            }
            /* Inhibitory */
#pragma omp for
            {
                for (int j = 0; j < nE_; j+=4){
                    float32x4_t I_rest = vdupq_n_f32(v_rest_I);
                    float32x4_t I_vE_I = vdupq_n_f32(vE_I);
                    
                    if (i<3) {
                        if (i == 0) {
                            verifying_I = {vaddv_f32(vld1_f32(&I_spike_previous->at(j*(time+1) + (time+1) - 3))) + I_spike_previous->at(j*(time+1) + (time+1)), vaddv_f32(vld1_f32(&I_spike_previous->at((j+1)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+1)*(time+1) + (time+1) - 1), vaddv_f32(vld1_f32(&I_spike_previous->at((j+2)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+2)*(time+1) + (time+1) - 1), vaddv_f32(vld1_f32(&I_spike_previous->at((j+3)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+3)*(time+1) + (time+1) - 1)};
                        }
                        else {
                            for (int k = 3; k > 0; --k) {
                                verifying_I[0] = verifying_I[0] - I_spike_previous->at(j*(time+1) + (time+1) - k) + I_spike_-> at(j*time + time - 3 - k);
                                verifying_I[1] = verifying_I[1] - I_spike_previous->at((j+1)*(time+1) + (time+1) - k) + I_spike_-> at((j+1)*(time+1) + (time+1) - 3 - k);
                                verifying_I[2] = verifying_I[2] - I_spike_previous->at((j+2)*(time+1) + (time+1) - k) + I_spike_-> at((j+2)*(time+1) + (time+1) - 3 - k);
                                verifying_I[3] = verifying_I[3] - I_spike_previous->at((j+3)*(time+1) + (time+1) - k) + I_spike_-> at((j+3)*(time+1) + (time+1) - 3 - k);
                            }
                        }
                        float32x4_t I_Potential4 = vld1q_f32(&(I_potential_->at(j)));
                        
                        /* E_to_I_spike_data = weight_S_E * Sensory_spike[:,i] */
                        /* Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max */
                        float32x4_t E_spike4 = {E_spike_->at(j*350 + i), E_spike_->at((j+1)*350 + i), E_spike_->at((j+2)*350 + i), E_spike_->at((j+3)*350 + i)};
                        float32x4_t I_Con_E4 = vld1q_f32(&(I_Con_E_->at(j)));
                        I_Con_E4 = vmulq_n_f32(I_Con_E4, gE_constant);
                        if (vaddvq_f32(E_spike4) != 0.0f) {
                            float32x4_t weight_E_I4 = vld1q_f32(&(E_I_weight_->at(j)));
                            float32x4_t E_I_spike_data = vmulq_f32(weight_E_I4, E_spike4);
                            float32x4_t I_dCon_E4 = vld1q_f32(&(I_dCon_E_->at(j)));
                            E_I_spike_data = vmulq_f32(E_I_spike_data, I_dCon_E4);
                            I_Con_E4 = vaddq_f32(I_Con_E4, E_I_spike_data);
                        }
                        vst1q_f32(&(I_Con_E_->at(j)), I_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Inter_dv_I = (-(Inter_potential_I - v_rest_i) - (Inter_I_gE/gL)*(Inter_potential_I-vE_I))*(time_step /tau_i) */
                        
                        I_rest = vsubq_f32(I_rest, I_Potential4);
                        
                        
                        I_vE_I = vsubq_f32(I_vE_I, I_Potential4);
                        I_Con_E4 = vdivq_f32(I_Con_E4, gL4);
                        
                        float32x4_t I_dPotential = vmlaq_f32(I_rest, I_Con_E4, I_vE_I);
                        
                        I_dPotential = vmulq_n_f32(I_dPotential, I_time_con);
                        
                        I_Potential4 = vaddq_f32(I_Potential4, I_dPotential);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_I[k] != 0 and I_Potential4[k] > v_reset_I) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        I_Potential4 = vmaxq_f32(v_rest_I4, I_Potential4);
                        uint32x4_t I_spike4_1 = vcgtq_f32(I_Potential4, v_thresh_I4);
                        uint32x4_t I_spike4_int = vshrq_n_u32(I_spike4_1, 31);
                        float32x4_t I_spike4 = vcvtq_f32_u32(I_spike4_int);
                        
                        I_spike_->at(j*350 + i+1) = I_spike4[0];
                        I_spike_->at((j+1)*350 + i+1) = I_spike4[1];
                        I_spike_->at((j+2)*350 + i+1) = I_spike4[2];
                        I_spike_->at((j+3)*350 + i+1) = I_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (I_spike4[k] == 1) {
                                I_Potential4[k] = v_rest_I;
                            }
                        }
                        vst1q_f32(&(I_potential_->at(j)), I_Potential4);
                    }
                    else {
                        float32x4_t verifying_0 = vld1q_f32(&(I_spike_->at(j*350 + i-3)));
                        float32x4_t verifying_1 = vld1q_f32(&(I_spike_->at((j+1)*350 + i-3)));
                        float32x4_t verifying_2 = vld1q_f32(&(I_spike_->at((j+2)*350 + i-3)));
                        float32x4_t verifying_3 = vld1q_f32(&(I_spike_->at((j+3)*350 + i-3)));
                        verifying_0[4] = 0;
                        verifying_1[4] = 0;
                        verifying_2[4] = 0;
                        verifying_3[4] = 0;
                        float32x4_t I_Potential4 = vld1q_f32(&(I_potential_->at(j)));
                        verifying_I = {vaddvq_f32(verifying_0), vaddvq_f32(verifying_1), vaddvq_f32(verifying_2), vaddvq_f32(verifying_3)};
                        
                        /* E_to_I_spike_data = weight_S_E * Sensory_spike[:,i] */
                        /* Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max */
                        float32x4_t E_spike4 = {E_spike_->at(j*350 + i), E_spike_->at((j+1)*350 + i), E_spike_->at((j+2)*350 + i), E_spike_->at((j+3)*350 + i)};
                        float32x4_t I_Con_E4 = vld1q_f32(&(I_Con_E_->at(j)));
                        I_Con_E4 = vmulq_n_f32(I_Con_E4, gE_constant);
                        if (vaddvq_f32(E_spike4) != 0.0f) {
                            float32x4_t weight_E_I4 = vld1q_f32(&(E_I_weight_->at(j)));
                            float32x4_t E_I_spike_data = vmulq_f32(weight_E_I4, E_spike4);
                            float32x4_t I_dCon_E4 = vld1q_f32(&(I_dCon_E_->at(j)));
                            E_I_spike_data = vmulq_f32(E_I_spike_data, I_dCon_E4);
                            I_Con_E4 = vaddq_f32(I_Con_E4, E_I_spike_data);
                        }
                        vst1q_f32(&(I_Con_E_->at(j)), I_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Inter_dv_I = (-(Inter_potential_I - v_rest_i) - (Inter_I_gE/gL)*(Inter_potential_I-vE_I))*(time_step /tau_i) */
                        I_rest = vsubq_f32(I_rest, I_Potential4);
                        
                        
                        I_vE_I = vsubq_f32(I_vE_I, I_Potential4);
                        I_Con_E4 = vdivq_f32(I_Con_E4, gL4);
                        
                        float32x4_t I_dPotential = vmlaq_f32(I_rest, I_Con_E4, I_vE_I);
                        
                        I_dPotential = vmulq_n_f32(I_dPotential, I_time_con);
                        
                        I_Potential4 = vaddq_f32(I_Potential4, I_dPotential);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_I[k] != 0 and I_Potential4[k] > v_reset_I) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        I_Potential4 = vmaxq_f32(v_rest_I4, I_Potential4);
                        uint32x4_t I_spike4_1 = vcgtq_f32(I_Potential4, v_thresh_I4);
                        uint32x4_t I_spike4_int = vshrq_n_u32(I_spike4_1, 31);
                        float32x4_t I_spike4 = vcvtq_f32_u32(I_spike4_int);
                        
                        I_spike_->at(j*350 + i+1) = I_spike4[0];
                        I_spike_->at((j+1)*350 + i+1) = I_spike4[1];
                        I_spike_->at((j+2)*350 + i+1) = I_spike4[2];
                        I_spike_->at((j+3)*350 + i+1) = I_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (I_spike4[k] == 1) {
                                I_Potential4[k] = v_rest_I;
                            }
                        }
                        vst1q_f32(&(I_potential_->at(j)), I_Potential4);
                    }
                }
            }
        }
    }
}

void SNN::process_data(vector<float> *neuron_index_num, int label, int &performance_count, int iter, int total_data, int epoch){
    int nE_ = this->nE_;
    float temp = 0;
    index_num->at(iter) = (float)label;
    vector<float> num_assignment(10,0);
    vector<float> summed_rate(10,0);
#pragma omp parallel
    {
        vector<float> num_assignment_private(10,0);
        vector<float> summed_rate_private(10,0);
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if(E_spike_total_->at(i)!= 0){
                    neuron_index_num ->at(i*total_data_*epoch_+iter) = E_spike_total_->at(i);
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                for (int j = 0; j < 10; ++j) {
                    if (neuron_index_->at(i) == j) {
                        num_assignment_private.at(j) += 1;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int j = 0; j < 10; ++j) {
                num_assignment.at(j) += num_assignment_private.at(j);
            }
        }
#pragma omp barrier
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (neuron_index_num ->at(i*total_data_*epoch_+iter) !=0) {
                    for (int j = 0; j < 10; ++j) {
                        if (neuron_index_->at(i) == j) {
                            summed_rate_private.at(j) += neuron_index_num ->at(i*total_data_*epoch_+iter)/num_assignment.at(j);
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int j = 0; j < 10; ++j) {
                summed_rate.at(j) += summed_rate_private.at(j);
            }
        }
    }
    for (int j = 0; j < 10; ++j) {
        if (summed_rate.at(j) > temp) {
            temp = summed_rate.at(j);
        }
    }
    for (int j = 0; j < 10; ++j) {
        if (summed_rate.at(j) == temp) {
            temp = j;
            break;
        }
    }
    if (label == (int)temp) {
        performance_count += 1;
    }
}

void SNN::set_index(int train_gap, int train_step){
    vector<float>temp(10,0);
    vector<float>neuron_total_spikes(nE_*10,0);
#pragma omp parallel
    {
        float32x4_t zero4 = vdupq_n_f32(0);
        vector<float>temp_private(10,0);
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                for (int j = train_gap*train_step; j<train_gap*(train_step+1); ++j) {
                    for (int k = 0; k<10; ++k) {
                        if (index_num->at(j)==(float)k) {
                            temp_private.at(k) +=1;
                            if (neuron_index_num_->at(i*total_data_*epoch_+j) != 0) {
                                neuron_total_spikes.at(i*10+k) += neuron_index_num_ ->at(i*total_data_*epoch_+j);
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int k = 0; k<10; ++k) {
                temp.at(k) += temp_private.at(k);
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                for (int j=0 ; j<10; j+=2) {
                    float32x2_t spike2_N = vld1_f32(&neuron_total_spikes.at(i*10+j));
                    float32x2_t spike2 = vld1_f32(&temp.at(j));
                    float32x2_t rate = vdiv_f32(spike2_N, spike2);
                    vst1_f32(&neuron_total_spikes.at(i*10+j), rate);
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                float big = 0;
                for (int j=0 ; j<10; ++j) {
                    if (neuron_total_spikes.at(i*10+j) > big) {
                        big = neuron_total_spikes.at(i*10+j);
                    }
                }
                for (int j=0 ; j<10; ++j) {
                    if (neuron_total_spikes.at(i*10+j) == big) {
                        neuron_index_ -> at(i) = j;
                    }
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                for (int j = train_gap*train_step; j<train_gap*(train_step+1); j+=4) {
                    vst1q_f32(&(neuron_index_num_ ->at(i*total_data_*epoch_+j)), zero4);
                }
            }
        }
    }
}

void SNN::STDP(int time, vector<float> *rate_dev){
    int nE_ = this->nE_;
    int nInput_ = this->nInput_;
#pragma omp parallel
    {
        vector<int> post_spike;
        vector<int> pre_spike;
        vector<int>::iterator iter;
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (rate_dev ->at(i) < 1 and E_spike_total_->at(i) != 0) {
                    learn ->at(i) *= rate_dev ->at(i);
                }
                /*else if (rate_dev ->at(i) < 1 and E_spike_total_->at(i) == 0) {
                    learn ->at(i) += 0.001*(1-learn ->at(i));
                }*/
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (E_spike_total_->at(i) != 0) {
                    for (int j = 0 ; j < time ; ++j) {
                        if (j == 0) {
                            post_spike.emplace_back(j);
                        }
                        else if (E_spike_->at(i*time+j) == 1) {
                            post_spike.emplace_back(j);
                        }
                    }
                    if (post_spike.size() > 1) {
                        for (int k = 0; k < nInput_ ; ++k){
                            for (iter = post_spike.begin(); iter != post_spike.end(); ++iter) {
                                if (iter != (post_spike.end()-1)) {
                                    for (int m = *iter; m<*(iter+1); ++m) {
                                        if (train_data_temp->at(m*nInput_+k) == 1) {
                                            pre_spike.emplace_back(m);
                                        }
                                    }
                                    if (pre_spike.size() != 0) {
                                        for (int l = 0; l<pre_spike.size(); ++l) {
                                            In_E_weight_->at(k*nE_+i) += learn ->at(i) * 0.01 * exp(-1*(float)(*(iter+1) - pre_spike.at(l))/20)* pow(1-In_E_weight_->at(k*nE_+i), 0.9);
                                        }
                                    }
                                }
                                else if(iter == (post_spike.end()-1) and *(post_spike.end()-1) != 349) {
                                    for (int m = *iter; m<time; ++m) {
                                        if (train_data_temp->at(m*nInput_+k) == 1) {
                                            pre_spike.emplace_back(m);
                                        }
                                    }
                                    if (pre_spike.size() != 0) {
                                        for (int l = 0; l<pre_spike.size(); ++l) {
                                            float mm = learn ->at(i) * 0.01 * exp((float)(*iter - pre_spike.at(l))/40)*pow(In_E_weight_->at(k*nE_+i),0.9);
                                            if (In_E_weight_->at(k*nE_+i) < mm) {
                                                In_E_weight_->at(k*nE_+i) = 0;
                                            }
                                            else {
                                                In_E_weight_->at(k*nE_+i) -= mm;
                                            }
                                        }
                                    }
                                }
                                pre_spike.clear();
                            }
                        }
                    }
                    post_spike.clear();
                }
            }
        }
    }
}

void SNN::resting(int time){
    float gE_constant = (1 - time_step / tau_syn_E);
    float gI_constant = (1 - time_step / tau_syn_I);
    float E_time_con = time_step/tau_E;
    float I_time_con = time_step/tau_I;
    int nE_ = this->nE_;
    int nI_ = this->nI_;
    int nInput_ = this->nInput_;
    vector<float> I_data_sum(nE_, 0);
    vector<float> total(nE_,0);
    vector<float> temp(nE_,0);
    vector<float> In_E_weight_copy_fired(nInput_*nE_);
    vector<float> E_dCon_E_copy_fired(nE_);
    vector<float> In_E_weight_copy_unfired(nInput_*nE_);
    vector<float> E_dCon_E_copy_unfired(nE_);
    float32x4_t temp4 = vdupq_n_f32(0.0f);
    float32x4_t gL4 = vdupq_n_f32(gL);
    float32x4_t v_thresh_E4 = vdupq_n_f32(v_thresh_E);
    float32x4_t v_thresh_I4 = vdupq_n_f32(v_thresh_I);
    float32x4_t v_reset_E4 = vdupq_n_f32(v_reset_E);
    float32x4_t v_reset_I4 = vdupq_n_f32(v_reset_I);
    float32x4_t v_rest_E4 = vdupq_n_f32(v_rest_E);
    float32x4_t v_rest_I4 = vdupq_n_f32(v_rest_I);
#pragma omp parallel
    {
        float32x4_t verifying_E = temp4;
        float32x4_t verifying_I = temp4;
        for (int i = 0; i < time; ++i) {
#pragma omp for
            {
                for (int j = 0; j < nE_; j+=4){
                    float32x4_t E_rest = vdupq_n_f32(v_rest_E);
                    float32x4_t E_vE_E = vdupq_n_f32(vE_E);
                    float32x4_t E_vI_E = vdupq_n_f32(vI_E);
                    
                    if (i<5) {
                        float32x4_t E_Potential4 = vld1q_f32(&(E_potential_->at(j)));
                        if (i == 0) {
                            verifying_E = {vaddvq_f32(vld1q_f32(&E_spike_previous->at(j*time + time - 5))) + E_spike_previous->at(j*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+1)*time + time - 5))) + E_spike_previous->at((j+1)*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+2)*time + time - 5))) + E_spike_previous->at((j+2)*time + time - 1), vaddvq_f32(vld1q_f32(&E_spike_previous->at((j+3)*time + time - 5))) + E_spike_previous->at((j+3)*time + time - 1)};
                        }
                        else {
                            for (int k = 5; k > 0; --k) {
                                verifying_E[0] = verifying_E[0] - E_spike_previous->at(j*time + time - k) + E_spike_-> at(j*time + time - 5 - k);
                                verifying_E[1] = verifying_E[1] - E_spike_previous->at((j+1)*time + time - k) + E_spike_-> at((j+1)*time + time - 5 - k);
                                verifying_E[2] = verifying_E[2] - E_spike_previous->at((j+2)*time + time - k) + E_spike_-> at((j+2)*time + time - 5 - k);
                                verifying_E[3] = verifying_E[3] - E_spike_previous->at((j+3)*time + time - k) + E_spike_-> at((j+3)*time + time - 5 - k);
                            }
                        }
                        
                        /* Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max */
                        
                        float32x4_t E_Con_E4 = vld1q_f32(&(E_Con_E_->at(j)));
                        float32x4_t input_data4;
                        float32x4_t E_dCon_E4;
                        float32x4_t Input_E_spike_data;
                        input_data4 = temp4;
                        E_Con_E4 = vmulq_n_f32(E_Con_E4, gE_constant);
                        E_dCon_E4 = vld1q_f32(&(E_dCon_E_->at(j)));
                        Input_E_spike_data = vmulq_f32(E_dCon_E4, input_data4);
                        E_Con_E4 = vaddq_f32(E_Con_E4, Input_E_spike_data);
                        vst1q_f32(&(E_Con_E_->at(j)), E_Con_E4);
                        
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /* Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                        float32x4_t I_E_spike_data;
                        for (int k = 0; k < nI_; ++k) {
                            if (I_spike_->at(k*time+i) == 1) {
                                float32x4_t weight_I_E4 = {I_E_weight_->at(k*nE_+j), I_E_weight_->at(k*nE_+j+1), I_E_weight_->at(k*nE_+j+2), I_E_weight_->at(k*nE_+j+3)};
                                float32x4_t I_spike4 = vdupq_n_f32(I_spike_->at(k*time+i));
                                I_E_spike_data = vmulq_f32(weight_I_E4, I_spike4);
                                I_data_sum.at(j) += I_E_spike_data[0];
                                I_data_sum.at(j+1) += I_E_spike_data[1];
                                I_data_sum.at(j+2) += I_E_spike_data[2];
                                I_data_sum.at(j+3) += I_E_spike_data[3];
                            }
                        }
                        float32x4_t E_dCon_I4 = vld1q_f32(&(E_dCon_I_->at(j)));
                        I_E_spike_data = vld1q_f32(&(I_data_sum.at(j)));
                        I_E_spike_data = vmulq_f32(I_E_spike_data, E_dCon_I4);
                        
                        float32x4_t E_Con_I4 = vld1q_f32(&(E_Con_I_->at(j)));
                        E_Con_I4 = vmulq_n_f32(E_Con_I4, gI_constant);
                        
                        E_Con_I4 = vaddq_f32(E_Con_I4, I_E_spike_data);
                        vst1q_f32(&(E_Con_I_->at(j)), E_Con_I4);
                        vst1q_f32(&(I_data_sum.at(j)), temp4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step /tau_e) */
                        
                        E_rest = vsubq_f32(E_rest, E_Potential4);
                        
                        E_vE_E = vsubq_f32(E_vE_E, E_Potential4);
                        E_Con_E4 = vdivq_f32(E_Con_E4, gL4);
                        
                        E_vI_E = vsubq_f32(E_vI_E, E_Potential4);
                        
                        E_Con_I4 = vdivq_f32(E_Con_I4, gL4);
                        
                        float32x4_t E_dPotential = vmlaq_f32(E_rest, E_Con_E4, E_vE_E);
                        E_dPotential = vmlaq_f32(E_dPotential, E_Con_I4, E_vI_E);
                        
                        E_dPotential = vmulq_n_f32(E_dPotential, E_time_con);
                        E_Potential4 = vaddq_f32(E_Potential4, E_dPotential);
                        E_Potential4 = vmaxq_f32(v_reset_E4, E_Potential4);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_E[k] != 0 and E_Potential4[k] > v_rest_E) {
                                E_Potential4[k] = v_rest_E;
                            }
                        }
                        uint32x4_t E_spike4_1 = vcgtq_f32(E_Potential4, v_thresh_E4);
                        uint32x4_t E_spike4_int = vshrq_n_u32(E_spike4_1, 31);
                        float32x4_t E_spike4 = vcvtq_f32_u32(E_spike4_int);
                        E_spike_->at(j*350 + i) = E_spike4[0];
                        E_spike_->at((j+1)*350 + i) = E_spike4[1];
                        E_spike_->at((j+2)*350 + i) = E_spike4[2];
                        E_spike_->at((j+3)*350 + i) = E_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (E_spike4[k] == 1) {
                                E_Potential4[k] = v_reset_E;
                            }
                        }
                        vst1q_f32(&(E_potential_->at(j)), E_Potential4);
                        /* END */
                    }
                    else{
                        float32x4_t verifying_0 = vld1q_f32(&(E_spike_->at(j*350 + i-5)));
                        float32x4_t verifying_1 = vld1q_f32(&(E_spike_->at((j+1)*350 + i-5)));
                        float32x4_t verifying_2 = vld1q_f32(&(E_spike_->at((j+2)*350 + i-5)));
                        float32x4_t verifying_3 = vld1q_f32(&(E_spike_->at((j+3)*350 + i-5)));
                        verifying_E = {vaddvq_f32(verifying_0) + E_spike_->at(j*350 + i-1), vaddvq_f32(verifying_1) + E_spike_->at((j+1)*350 + i-1), vaddvq_f32(verifying_2) + E_spike_->at((j+2)*350 + i-1), vaddvq_f32(verifying_3) + E_spike_->at((j+3)*350 + i-1)};
                        float32x4_t E_Potential4 = vld1q_f32(&(E_potential_->at(j)));
                        
                        /* Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max */
                        float32x4_t E_Con_E4 = vld1q_f32(&(E_Con_E_->at(j)));
                        float32x4_t input_data4;
                        float32x4_t E_dCon_E4;
                        float32x4_t Input_E_spike_data;
                        input_data4 = temp4;
                        E_Con_E4 = vmulq_n_f32(E_Con_E4, gE_constant);
                        E_dCon_E4 = vld1q_f32(&(E_dCon_E_->at(j)));
                        Input_E_spike_data = vmulq_f32(E_dCon_E4, input_data4);
                        E_Con_E4 = vaddq_f32(E_Con_E4, Input_E_spike_data);
                        vst1q_f32(&(E_Con_E_->at(j)), E_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /* Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                        float32x4_t I_E_spike_data;
                        for (int k = 0; k < nI_; ++k) {
                            if (I_spike_->at(k*time+i) == 1) {
                                float32x4_t weight_I_E4 = {I_E_weight_->at(k*nE_+j), I_E_weight_->at(k*nE_+j+1), I_E_weight_->at(k*nE_+j+2), I_E_weight_->at(k*nE_+j+3)};
                                float32x4_t I_spike4 = vdupq_n_f32(I_spike_->at(k*time+i));
                                I_E_spike_data = vmulq_f32(weight_I_E4, I_spike4);
                                I_data_sum.at(j) += I_E_spike_data[0];
                                I_data_sum.at(j+1) += I_E_spike_data[1];
                                I_data_sum.at(j+2) += I_E_spike_data[2];
                                I_data_sum.at(j+3) += I_E_spike_data[3];
                            }
                        }
                        float32x4_t E_dCon_I4 = vld1q_f32(&(E_dCon_I_->at(j)));
                        I_E_spike_data = vld1q_f32(&(I_data_sum.at(j)));
                        I_E_spike_data = vmulq_f32(I_E_spike_data, E_dCon_I4);
                        
                        float32x4_t E_Con_I4 = vld1q_f32(&(E_Con_I_->at(j)));
                        E_Con_I4 = vmulq_n_f32(E_Con_I4, gI_constant);
                        
                        E_Con_I4 = vaddq_f32(E_Con_I4, I_E_spike_data);
                        vst1q_f32(&(E_Con_I_->at(j)), E_Con_I4);
                        vst1q_f32(&(I_data_sum.at(j)), temp4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step /tau_e) */
                        E_rest = vsubq_f32(E_rest, E_Potential4);
                        
                        E_vE_E = vsubq_f32(E_vE_E, E_Potential4);
                        E_Con_E4 = vdivq_f32(E_Con_E4, gL4);
                        
                        E_vI_E = vsubq_f32(E_vI_E, E_Potential4);
                        E_Con_I4 = vdivq_f32(E_Con_I4, gL4);
                        
                        float32x4_t E_dPotential = vmlaq_f32(E_rest, E_Con_E4, E_vE_E);
                        E_dPotential = vmlaq_f32(E_dPotential, E_Con_I4, E_vI_E);
                        
                        E_dPotential = vmulq_n_f32(E_dPotential, E_time_con);
                        E_Potential4 = vaddq_f32(E_Potential4, E_dPotential);
                        E_Potential4 = vmaxq_f32(v_reset_E4, E_Potential4);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_E[k] != 0 and E_Potential4[k] > v_rest_E) {
                                E_Potential4[k] = v_rest_E;
                            }
                        }
                        uint32x4_t E_spike4_1 = vcgtq_f32(E_Potential4, v_thresh_E4);
                        uint32x4_t E_spike4_int = vshrq_n_u32(E_spike4_1, 31);
                        float32x4_t E_spike4 = vcvtq_f32_u32(E_spike4_int);
                        E_spike_->at(j*350 + i) = E_spike4[0];
                        E_spike_->at((j+1)*350 + i) = E_spike4[1];
                        E_spike_->at((j+2)*350 + i) = E_spike4[2];
                        E_spike_->at((j+3)*350 + i) = E_spike4[3];
                        
                        for (int k = 0; k < 4; ++k) {
                            if (E_spike4[k] == 1) {
                                E_Potential4[k] = v_reset_E;
                            }
                        }
                        vst1q_f32(&(E_potential_->at(j)), E_Potential4);
                        /* end */
                    }
                }
            }
            /* Inhibitory */
#pragma omp for
            {
                for (int j = 0; j < nE_; j+=4){
                    float32x4_t I_rest = vdupq_n_f32(v_rest_I);
                    float32x4_t I_vE_I = vdupq_n_f32(vE_I);
                    
                    if (i<3) {
                        if (i == 0) {
                            verifying_I = {vaddv_f32(vld1_f32(&I_spike_previous->at(j*(time+1) + (time+1) - 3))) + I_spike_previous->at(j*(time+1) + (time+1)), vaddv_f32(vld1_f32(&I_spike_previous->at((j+1)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+1)*(time+1) + (time+1) - 1), vaddv_f32(vld1_f32(&I_spike_previous->at((j+2)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+2)*(time+1) + (time+1) - 1), vaddv_f32(vld1_f32(&I_spike_previous->at((j+3)*(time+1) + (time+1) - 3))) + I_spike_previous->at((j+3)*(time+1) + (time+1) - 1)};
                        }
                        else {
                            for (int k = 3; k > 0; --k) {
                                verifying_I[0] = verifying_I[0] - I_spike_previous->at(j*(time+1) + (time+1) - k) + I_spike_-> at(j*time + time - 3 - k);
                                verifying_I[1] = verifying_I[1] - I_spike_previous->at((j+1)*(time+1) + (time+1) - k) + I_spike_-> at((j+1)*(time+1) + (time+1) - 3 - k);
                                verifying_I[2] = verifying_I[2] - I_spike_previous->at((j+2)*(time+1) + (time+1) - k) + I_spike_-> at((j+2)*(time+1) + (time+1) - 3 - k);
                                verifying_I[3] = verifying_I[3] - I_spike_previous->at((j+3)*(time+1) + (time+1) - k) + I_spike_-> at((j+3)*(time+1) + (time+1) - 3 - k);
                            }
                        }
                        float32x4_t I_Potential4 = vld1q_f32(&(I_potential_->at(j)));
                        
                        /* E_to_I_spike_data = weight_S_E * Sensory_spike[:,i] */
                        /* Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max */
                        float32x4_t E_spike4 = {E_spike_->at(j*350 + i), E_spike_->at((j+1)*350 + i), E_spike_->at((j+2)*350 + i), E_spike_->at((j+3)*350 + i)};
                        float32x4_t I_Con_E4 = vld1q_f32(&(I_Con_E_->at(j)));
                        I_Con_E4 = vmulq_n_f32(I_Con_E4, gE_constant);
                        if (vaddvq_f32(E_spike4) != 0.0f) {
                            float32x4_t weight_E_I4 = vld1q_f32(&(E_I_weight_->at(j)));
                            float32x4_t E_I_spike_data = vmulq_f32(weight_E_I4, E_spike4);
                            float32x4_t I_dCon_E4 = vld1q_f32(&(I_dCon_E_->at(j)));
                            E_I_spike_data = vmulq_f32(E_I_spike_data, I_dCon_E4);
                            I_Con_E4 = vaddq_f32(I_Con_E4, E_I_spike_data);
                        }
                        vst1q_f32(&(I_Con_E_->at(j)), I_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Inter_dv_I = (-(Inter_potential_I - v_rest_i) - (Inter_I_gE/gL)*(Inter_potential_I-vE_I))*(time_step /tau_i) */
                        
                        I_rest = vsubq_f32(I_rest, I_Potential4);
                        
                        
                        I_vE_I = vsubq_f32(I_vE_I, I_Potential4);
                        I_Con_E4 = vdivq_f32(I_Con_E4, gL4);
                        
                        float32x4_t I_dPotential = vmlaq_f32(I_rest, I_Con_E4, I_vE_I);
                        
                        I_dPotential = vmulq_n_f32(I_dPotential, I_time_con);
                        
                        I_Potential4 = vaddq_f32(I_Potential4, I_dPotential);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_I[k] != 0 and I_Potential4[k] > v_reset_I) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        I_Potential4 = vmaxq_f32(v_rest_I4, I_Potential4);
                        uint32x4_t I_spike4_1 = vcgtq_f32(I_Potential4, v_thresh_I4);
                        uint32x4_t I_spike4_int = vshrq_n_u32(I_spike4_1, 31);
                        float32x4_t I_spike4 = vcvtq_f32_u32(I_spike4_int);
                        I_spike_->at(j*350 + i+1) = I_spike4[0];
                        I_spike_->at((j+1)*350 + i+1) = I_spike4[1];
                        I_spike_->at((j+2)*350 + i+1) = I_spike4[2];
                        I_spike_->at((j+3)*350 + i+1) = I_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (I_spike4[k] == 1) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        vst1q_f32(&(I_potential_->at(j)), I_Potential4);
                    }
                    else {
                        float32x4_t verifying_0 = vld1q_f32(&(I_spike_->at(j*350 + i-3)));
                        float32x4_t verifying_1 = vld1q_f32(&(I_spike_->at((j+1)*350 + i-3)));
                        float32x4_t verifying_2 = vld1q_f32(&(I_spike_->at((j+2)*350 + i-3)));
                        float32x4_t verifying_3 = vld1q_f32(&(I_spike_->at((j+3)*350 + i-3)));
                        verifying_0[4] = 0;
                        verifying_1[4] = 0;
                        verifying_2[4] = 0;
                        verifying_3[4] = 0;
                        float32x4_t I_Potential4 = vld1q_f32(&(I_potential_->at(j)));
                        verifying_I = {vaddvq_f32(verifying_0), vaddvq_f32(verifying_1), vaddvq_f32(verifying_2), vaddvq_f32(verifying_3)};
                        
                        /* E_to_I_spike_data = weight_S_E * Sensory_spike[:,i] */
                        /* Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max */
                        float32x4_t E_spike4 = {E_spike_->at(j*350 + i), E_spike_->at((j+1)*350 + i), E_spike_->at((j+2)*350 + i), E_spike_->at((j+3)*350 + i)};
                        float32x4_t I_Con_E4 = vld1q_f32(&(I_Con_E_->at(j)));
                        I_Con_E4 = vmulq_n_f32(I_Con_E4, gE_constant);
                        if (vaddvq_f32(E_spike4) != 0.0f) {
                            float32x4_t weight_E_I4 = vld1q_f32(&(E_I_weight_->at(j)));
                            float32x4_t E_I_spike_data = vmulq_f32(weight_E_I4, E_spike4);
                            float32x4_t I_dCon_E4 = vld1q_f32(&(I_dCon_E_->at(j)));
                            E_I_spike_data = vmulq_f32(E_I_spike_data, I_dCon_E4);
                            I_Con_E4 = vaddq_f32(I_Con_E4, E_I_spike_data);
                        }
                        vst1q_f32(&(I_Con_E_->at(j)), I_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Inter_dv_I = (-(Inter_potential_I - v_rest_i) - (Inter_I_gE/gL)*(Inter_potential_I-vE_I))*(time_step /tau_i) */
                        I_rest = vsubq_f32(I_rest, I_Potential4);
                        
                        
                        I_vE_I = vsubq_f32(I_vE_I, I_Potential4);
                        I_Con_E4 = vdivq_f32(I_Con_E4, gL4);
                        
                        float32x4_t I_dPotential = vmlaq_f32(I_rest, I_Con_E4, I_vE_I);
                        
                        I_dPotential = vmulq_n_f32(I_dPotential, I_time_con);
                        
                        I_Potential4 = vaddq_f32(I_Potential4, I_dPotential);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_I[k] != 0 and I_Potential4[k] > v_reset_I) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        I_Potential4 = vmaxq_f32(v_rest_I4, I_Potential4);
                        uint32x4_t I_spike4_1 = vcgtq_f32(I_Potential4, v_thresh_I4);
                        uint32x4_t I_spike4_int = vshrq_n_u32(I_spike4_1, 31);
                        float32x4_t I_spike4 = vcvtq_f32_u32(I_spike4_int);
                        I_spike_->at(j*350 + i+1) = I_spike4[0];
                        I_spike_->at((j+1)*350 + i+1) = I_spike4[1];
                        I_spike_->at((j+2)*350 + i+1) = I_spike4[2];
                        I_spike_->at((j+3)*350 + i+1) = I_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (I_spike4[k] == 1) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        vst1q_f32(&(I_potential_->at(j)), I_Potential4);
                    }
                }
            }
        }
    }
}
