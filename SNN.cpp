//
//  SNN.cpp
//  SNN
//
//  Created by 전민기 on 2022/12/26.
//

#include "SNN.hpp"
#include <iostream>
#include <vector>
#include <cstring>  
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
#include <limits>

void SNN::set_initial(int iteration, double interval){
    this->iteration_ = iteration;
    this->interval_ = interval;
}

void SNN::initializatoin(int &E_total_spike, int time, int iteration){
    int nE_ = this->nE_;
#pragma omp parallel
    {
        float64x2_t zero2 = vdupq_n_f64(0);
        int32x4_t zero4 = vdupq_n_s32(0);
#pragma omp for
        {
            for (int i = 0; i < nE_; i+=4) {
                vst1q_s32(&(E_spike_total_ptr[i]), zero4);
                /*if (neuron_exclusion == nE_) {
                    float64x2_t E_Potential4 = vld1q_f64(&E_potential_ptr[i]);
                    E_Potential4 = vminq_f64(vdupq_n_f64(-0.065), E_Potential4);
                    vst1q_f64(&E_potential_ptr[i], E_Potential4);
                }*/
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_*time; i+=2) {;
                vst1q_f64(&(E_potential_capacitance_ptr[i]), zero2);
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nInput_*time; i+=4) {
                vst1q_s32(&train_data_temp_ptr[i], zero4);
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_*time; i+=4) {
                vst1q_s32(&E_spike_ptr[i], zero4);
                vst1q_s32(&input_data_copy_ptr[i], zero4);
            }
        }
    }
    E_total_spike = 0;
}

void SNN::Stimulation(int time, vector<double> &after_save, vector<int> &E_spike_total, int &E_total_spike, double* weight_check_ptr, int train_label){
    //double I_time_con = time_step_/tau_I;
    int nE_ = this->nE_;
    int nInput_ = this->nInput_;
    int interation_ = this->iteration_;
    double interval_ = this->interval_;
    double recover_constant = (double)time*train_gap_;
    vector<double> prior(nE_);
    double* prior_ptr = prior.data();
    vector<double> latter(nE_);
    double* latter_ptr = latter.data();
    float64x2_t temp2 = vdupq_n_f64(0.0);
    float64x2_t gL4 = vdupq_n_f64(gL);
    vector<vector<int>> time_check(nE_);
    //struct timespec begin, end ;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0f, 1.0f);
    vector<double> weight_in_E_(nE_,0);
    vector<double> input_rate(nInput_,0);
    double* input_rate_ptr = input_rate.data();
    vector<int> adaptation(nE_,0);
    int* adaptation_ptr = adaptation.data();
    float64x2_t E_Con_E4 = temp2;
    float64x2_t input_data4 = temp2;
    float64x2_t I_E_spike_data = temp2;
    float64x2_t E_Con_I4 = temp2;
    float64x2_t E_Potential4 = temp2;
    float64x2_t weight_temp2 = temp2;
    int32x2_t verifying_E4 = vdup_n_s32(0);
    float64x2_t one2 = vdupq_n_f64(1);
    float64x2_t computation_admit = temp2;
    float64x2_t E_rest1 = temp2;
    float64x2_t E_vE_E1 = temp2;
    float64x2_t E_dPotential = temp2;
    float64x2_t Input_E_spike_data = temp2;
    float64x2_t E_vI_E1 = temp2;
    double E_Potential4k = 0;
    int lasttime = time - 1;
    double gE_constant = (1 - time_step_ / tau_syn_E);
    double gI_constant = (1 - time_step_ / tau_syn_I);
    double E_time_con = time_step_/tau_E;
    float64x2_t E_rest = vdupq_n_f64(v_rest_E);
    float64x2_t E_vE_E = vdupq_n_f64(vE_E);
    float64x2_t E_vI_E = vdupq_n_f64(vI_E);
    double* weight_in_E_ptr = weight_in_E_.data();
    double* after_save_prt = after_save.data();
    char weight_check_In_E = 0;
    char weight_check_I_E = 0;
    vector<int> post_spike;
    vector<int> pre_spike;
    vector<int> over_vth;
    vector<double> spikes(nE_,1);
    double* spikes_ptr = spikes.data();
    vector<int> train_assign;
    for (int j=0; j<nInput_; ++j)  {
        if (weight_check_ptr[j*10+train_label] > 1) {
            train_assign.emplace_back(j);
        }
    }
#pragma omp parallel
    {
        double a = time_step_ / 8 * interval_;
#pragma omp for
        {
            for (int j=nInput_*interation_; j<nInput_*(interation_+1); j+=2)  {
                float64x2_t v1 = vld1q_f64(&train_data_ptr[j]);
                int input_neuron = j - (nInput_*interation_);
                if (vaddvq_f64(v1) != 0.0f) {
                    v1 = vmulq_n_f64(v1, a);
                    vst1q_f64(&input_rate_ptr[input_neuron], v1);
                }
            }
        }
    }
//#pragma omp parallel
    {
        for (int i = 0; i < time; ++i) {
            //double random_v = dis(gen);
            //#pragma omp for
            {
                for (int j=0; j<nInput_; ++j)  {
                    if (input_rate_ptr[j] != 0) {
                        if (input_rate_ptr[j] > dis(gen)) {
                            weight_check_In_E = 1;
                            train_data_temp_ptr[i*nInput_+j] = 1;
                            //#pragma omp critical
                            {
                                for (int k = 0; k < nE_; k+=2) {
                                    weight_temp2 = vld1q_f64(&In_E_weight_ptr[j*nE_ + k]);
                                    vst1q_f64(&weight_in_E_ptr[k], vaddq_f64(weight_temp2, vld1q_f64(&weight_in_E_ptr[k])));
                                }
                            }
                        }
                    }
                }
            }
            //#pragma omp for
            {
                for (int j = 0; j < nE_; j+=2) {
                    weight_check_I_E = 0;
                    E_Con_E4 = vld1q_f64(&(E_Con_E_ptr[j]));
                    E_Con_I4 = vld1q_f64(&(E_Con_I_ptr[j]));
                    E_Potential4 = vld1q_f64(&(E_potential_ptr[j]));
                    verifying_E4 = vld1_s32(&(verifying_E_ptr[j]));
                    if (i == 0) {
                        I_E_spike_data = vld1q_f64(&I_data_ptr[(time-1)*nE_+j]);
                        if (vaddvq_f64(I_E_spike_data) != 0) {
                            weight_check_I_E = 1;
                            vst1q_f64(&I_data_ptr[(time-1)*nE_+j], temp2);
                        }
                    }
                    else {
                        I_E_spike_data = vld1q_f64(&I_data_ptr[(i-1)*nE_+j]);
                        if (vaddvq_f64(I_E_spike_data) != 0) {
                            weight_check_I_E = 1;
                            vst1q_f64(&I_data_ptr[(i-1)*nE_+j], temp2);
                        }
                    }
                    if (vaddvq_f64(E_Potential4)/ 2 != -0.065 || vaddvq_f64(E_Con_E4) != 0 || weight_check_In_E != 0 || weight_check_I_E != 0 || vaddvq_f64(E_Con_I4) != 0) {
                        input_data4 = vld1q_f64(&weight_in_E_ptr[j]);
                        if (vaddv_s32(verifying_E4) == 0) {
                            computation_admit = one2;
                        }
                        else {
                            computation_admit = temp2;
                            for (int k = 0; k < 2; ++k) {
                                if(verifying_E4[k] == 0) {
                                    computation_admit[k] = 1;
                                }
                                else {
                                    if (input_data4[k] != 0) {
                                        input_data_copy_ptr[i*nE_+j+k] = -1;
                                    }
                                }
                            }
                        }
                        if (vaddvq_f64(computation_admit) == 0) {
                            for (int k = 0; k < 2; ++k) {
                                if(E_dCon_E_ptr[j+k] < 1) {
                                    E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                }
                                if (verifying_E4[k] < (int)refractory_e && verifying_E4[k] > 0) {
                                    verifying_E_ptr[j+k] += 1;
                                }
                                else if(verifying_E4[k] == (int)refractory_e && verifying_E4[k] > 0) {
                                    verifying_E_ptr[j+k] = 0;
                                }
                                if (input_data4[k] != 0) {
                                    input_data_copy_ptr[i*nE_+j+k] = -1;
                                }
                            }
                        }
                        else {
                            E_rest1 = vsubq_f64(E_rest, E_Potential4);
                            E_vE_E1 = vsubq_f64(E_vE_E, E_Potential4);
                            E_dPotential = E_rest1;
                            /* Sensory_gE = Sensory_gE * (1 - time_step_ / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max */
                            if (vaddvq_f64(input_data4) != 0) {
                                Input_E_spike_data = vmulq_f64(vld1q_f64(&E_dCon_E_ptr[j]), input_data4);
                                if(vaddvq_f64(E_Con_E4) != 0) {
                                    E_Con_E4 = vmulq_n_f64(E_Con_E4, gE_constant);
                                    E_Con_E4 = vaddq_f64(E_Con_E4, Input_E_spike_data);
                                    vst1q_f64(&(E_Con_E_ptr[j]), E_Con_E4);
                                    E_Con_E4 = vdivq_f64(E_Con_E4, gL4);
                                    E_dPotential = vmlaq_f64(E_rest1, E_Con_E4, E_vE_E1);
                                }
                                else {
                                    vst1q_f64(&(E_Con_E_ptr[j]), Input_E_spike_data);
                                    E_Con_E4 = vdivq_f64(Input_E_spike_data, gL4);
                                    E_dPotential = vmlaq_f64(E_rest1, E_Con_E4, E_vE_E1);
                                }
                                vst1q_f64(&weight_in_E_ptr[j], temp2);
                            }
                            else {
                                if(vaddvq_f64(E_Con_E4) != 0) {
                                    E_Con_E4 = vmulq_n_f64(E_Con_E4, gE_constant);
                                    vst1q_f64(&(E_Con_E_ptr[j]), E_Con_E4);
                                    E_Con_E4 = vdivq_f64(E_Con_E4, gL4);
                                    E_dPotential = vmlaq_f64(E_rest1, E_Con_E4, E_vE_E1);
                                }
                            }
                            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            
                            /* Sensory_gI = Sensory_gI * (1 - time_step_ / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                            E_vI_E1 = vsubq_f64(E_vI_E, E_Potential4);
                            if (vaddvq_f64(I_E_spike_data) != 0) {
                                I_E_spike_data = vmulq_f64(I_E_spike_data, vld1q_f64(&E_dCon_I_ptr[j]));
                                if (vaddvq_f64(E_Con_I4) != 0) {
                                    E_Con_I4 = vmulq_n_f64(E_Con_I4, gI_constant);
                                    E_Con_I4 = vaddq_f64(E_Con_I4, I_E_spike_data);
                                    vst1q_f64(&(E_Con_I_ptr[j]), E_Con_I4);
                                    E_Con_I4 = vdivq_f64(E_Con_I4, gL4);
                                    E_dPotential = vmlaq_f64(E_dPotential, E_Con_I4, E_vI_E1);
                                }
                                else {
                                    vst1q_f64(&(E_Con_I_ptr[j]), I_E_spike_data);
                                    E_Con_I4 = vdivq_f64(I_E_spike_data, gL4);
                                    E_dPotential = vmlaq_f64(E_dPotential, E_Con_I4, E_vI_E1);
                                }
                            }
                            else {
                                if (vaddvq_f64(E_Con_I4) != 0) {
                                    E_Con_I4 = vmulq_n_f64(E_Con_I4, gI_constant);
                                    vst1q_f64(&(E_Con_I_ptr[j]), E_Con_I4);
                                    E_Con_I4 = vdivq_f64(E_Con_I4, gL4);
                                    E_dPotential = vmlaq_f64(E_dPotential, E_Con_I4, E_vI_E1);
                                }
                            }
                            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            
                            /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step_ /tau_e) */
                            
                            E_dPotential = vmulq_n_f64(E_dPotential, E_time_con);
                            E_dPotential = vmulq_f64(E_dPotential, computation_admit);
                            E_Potential4 = vaddq_f64(E_Potential4, E_dPotential);
                            E_Potential4 = vmaxq_f64(E_vI_E, E_Potential4);
                            for (int k = 0; k < 2; ++k) {
                                E_Potential4k = E_Potential4[k];
                                int* time_checkjk_ptr = time_check[j+k].data();
                                if (E_Potential4k <= v_thresh_E) {
                                    if (E_potential_ptr[j+k]-E_Potential4k < 0) {
                                        if (input_data4[k] != 0) {
                                            time_check[j+k].emplace_back(i);
                                            E_potential_capacitance_ptr[(j+k)*time + i] = E_potential_ptr[j+k];
                                            input_data_copy_ptr[i*nE_+j+k] = 1;
                                        }
                                        if(i == lasttime) {
                                            if (E_spike_total_ptr[j+k] != 0) {
                                                double a_private = 0;
                                                for (int l = 0; l<nInput_; ++l) {
                                                    a_private += In_E_weight_ptr[l*nE_+(j+k)];
                                                }
                                                prior_ptr[j+k] = a_private;
                                                adaptation_ptr[j+k] = 1;
                                                post_spike.emplace_back(i);
                                                for (int l = lasttime ; l >= 0 ; --l) {
                                                    if (E_spike_ptr[(j+k)*time + l] == 1) {
                                                        post_spike.emplace_back(l);
                                                        break;
                                                    }
                                                    else if (l == 0) {
                                                        post_spike.emplace_back(l);
                                                        break;
                                                    }
                                                }
                                                for (int l = 0; l < nInput_ ; ++l){
                                                    int start = *(post_spike.end()-1);
                                                    int end = *post_spike.begin();
                                                    for (int m = start; m < end; ++m) {
                                                        if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+(j+k)] != 0) {
                                                            pre_spike.emplace_back(m);
                                                        }
                                                    }
                                                    int* pre_spike_ptr = pre_spike.data();
                                                    int pre_spike_size = (int)pre_spike.size();
                                                    if (pre_spike_size != 0) {
                                                        for (int m = 0; m<pre_spike_size; ++m) {
                                                            In_E_weight_ptr[l*nE_+(j+k)] -=  0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+(j+k)];
                                                        }
                                                    }
                                                    pre_spike.clear();
                                                }
                                                post_spike.clear();
                                                if (adaptation_ptr[j+k] == 1) {
                                                    double b_private = 0;
                                                    for (int l = 0; l<nInput_; ++l) {
                                                        b_private += In_E_weight_ptr[l*nE_+(j+k)];
                                                    }
                                                    latter_ptr[j+k] = b_private;
                                                    if (latter_ptr[j+k] - prior_ptr[j+k]<0) {
                                                        E_dCon_E_ptr[j+k] *= latter_ptr[j+k]/(prior_ptr[j+k]*spikes_ptr[j+k]);
                                                        after_save_prt[j+k] = (1-E_dCon_E_ptr[j+k])/recover_constant;
                                                    }
                                                    else if (latter_ptr[j+k] - prior_ptr[j+k]>0){
                                                        E_dCon_E_ptr[j+k] *= prior_ptr[j+k]/(latter_ptr[j+k]*spikes_ptr[j+k]);
                                                        after_save_prt[j+k] = (1-E_dCon_E_ptr[j+k])/recover_constant;
                                                    }
                                                    adaptation_ptr[j+k] = 0;
                                                }
                                            }
                                            else {
                                                if(E_dCon_E_ptr[j+k] < 1) {
                                                    E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                                }
                                            }
                                        }
                                        else {
                                            if(E_dCon_E_ptr[j+k] < 1) {
                                                E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                            }
                                        }
                                    }
                                    else if(E_potential_ptr[j+k]-E_Potential4k > 0) {
                                        if (input_data4[k] != 0) {
                                            input_data_copy_ptr[i*nE_+j+k] = -1;
                                        }
                                        int time_check_size =(int)time_check[j+k].size();
                                        if (time_check_size != 0) {
                                            int number = 0;
                                            for (int l = time_check_size-1; l >= 0; --l) {
                                                if (E_potential_capacitance_ptr[(j+k)*time + time_checkjk_ptr[l]]-E_Potential4k > 0) {
                                                    input_data_copy_ptr[time_checkjk_ptr[l]*nE_+j+k] = 0;
                                                    number += 1;
                                                }
                                                else {
                                                    break;
                                                }
                                            }
                                            if (number != 0) {
                                                for (int l = 0; l < number; ++l) {
                                                    time_check[j+k].pop_back();
                                                }
                                            }
                                        }
                                        if(i == lasttime) {
                                            if (E_spike_total_ptr[j+k] != 0) {
                                                double a_private = 0;
                                                for (int l = 0; l<nInput_; ++l) {
                                                    a_private += In_E_weight_ptr[l*nE_+(j+k)];
                                                }
                                                prior_ptr[j+k] = a_private;
                                                adaptation_ptr[j+k] = 1;
                                                post_spike.emplace_back(i);
                                                for (int l = lasttime; l >= 0 ; --l) {
                                                    if (E_spike_ptr[(j+k)*time + l] == 1) {
                                                        post_spike.emplace_back(l);
                                                        break;
                                                    }
                                                    else if (l == 0) {
                                                        post_spike.emplace_back(l);
                                                        break;
                                                    }
                                                }
                                                for (int l = 0; l < nInput_ ; ++l){
                                                    int start = *(post_spike.end()-1);
                                                    int end = *post_spike.begin();
                                                    for (int m = start; m < end; ++m) {
                                                        if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+(j+k)] != 0) {
                                                            pre_spike.emplace_back(m);
                                                        }
                                                    }
                                                    int* pre_spike_ptr = pre_spike.data();
                                                    int pre_spike_size = (int)pre_spike.size();
                                                    if (pre_spike_size != 0) {
                                                        for (int m = 0; m<pre_spike_size; ++m) {
                                                            In_E_weight_ptr[l*nE_+(j+k)] -= 0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+(j+k)];
                                                        }
                                                    }
                                                    pre_spike.clear();
                                                }
                                                post_spike.clear();
                                                if (adaptation_ptr[j+k] == 1) {
                                                    double b_private = 0;
                                                    for (int l = 0; l<nInput_; ++l) {
                                                        b_private += In_E_weight_ptr[l*nE_+(j+k)];
                                                    }
                                                    latter_ptr[j+k] = b_private;
                                                    if (latter_ptr[j+k] - prior_ptr[j+k]<0) {
                                                        E_dCon_E_ptr[j+k] *= latter_ptr[j+k]/(prior_ptr[j+k]*spikes_ptr[j+k]);
                                                        after_save_prt[j+k] = (1-E_dCon_E_ptr[j+k])/recover_constant;
                                                    }
                                                    else if (latter_ptr[j+k] - prior_ptr[j+k]>0){
                                                        E_dCon_E_ptr[j+k] *= prior_ptr[j+k]/(latter_ptr[j+k]*spikes_ptr[j+k]);
                                                        after_save_prt[j+k] = (1-E_dCon_E_ptr[j+k])/recover_constant;
                                                    }
                                                    adaptation_ptr[j+k] = 0;
                                                }
                                            }
                                            else {
                                                if(E_dCon_E_ptr[j+k] < 1) {
                                                    E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                                }
                                            }
                                        }
                                        else {
                                            if(E_dCon_E_ptr[j+k] < 1) {
                                                E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                            }
                                        }
                                        
                                    }
                                    else {
                                        if(E_dCon_E_ptr[j+k] < 1) {
                                            E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                                        }
                                        if (verifying_E4[k] < (int)refractory_e && verifying_E4[k] > 0) {
                                            verifying_E_ptr[j+k] += 1;
                                        }
                                        else if(verifying_E4[k] == (int)refractory_e && verifying_E4[k] > 0) {
                                            verifying_E_ptr[j+k] = 0;
                                        }
                                        if (input_data4[k] != 0) {
                                            input_data_copy_ptr[i*nE_+j+k] = -1;
                                        }
                                    }
                                }
                                else {
                                    over_vth.emplace_back(j+k);
                                }
                            }
                            vst1q_f64(&(E_potential_ptr[j]), E_Potential4);
                        }
                    }
                    else {
                        for(int k = 0; k < 2; ++k) {
                            if(E_dCon_E_ptr[j+k] < 1) {
                                E_dCon_E_ptr[j+k] += after_save_prt[j+k];
                            }
                            if (verifying_E4[k] < (int)refractory_e && verifying_E4[k] > 0) {
                                verifying_E_ptr[j+k] += 1;
                            }
                            else if(verifying_E4[k] == (int)refractory_e && verifying_E4[k] > 0) {
                                verifying_E_ptr[j+k] = 0;
                            }
                        }
                    }
                }
                int over_vth_size = (int)over_vth.size();
                int* over_vth_ptr = over_vth.data();
                int trained_neuron = 0;
                double potential = -1;
                for (int j = 0; j < over_vth_size; ++j) {
                    if (E_potential_ptr[over_vth_ptr[j]] > potential) {
                        potential = E_potential_ptr[over_vth_ptr[j]];
                    }
                }
                for (int j = 0; j < over_vth_size; ++j) {
                    if (E_potential_ptr[over_vth_ptr[j]] == potential) {
                        trained_neuron = over_vth_ptr[j];
                        break;
                    }
                }
                for (int j = 0; j < over_vth_size; ++j) {
                    if (over_vth_ptr[j] != trained_neuron) {
                        if (input_data4[over_vth_ptr[j]] != 0) {
                            time_check[over_vth_ptr[j]].emplace_back(i);
                            E_potential_capacitance_ptr[over_vth_ptr[j]*time + i]= E_potential_ptr[over_vth_ptr[j]];
                            input_data_copy_ptr[i*nE_+over_vth_ptr[j]] = 1;
                        }
                        if(i == lasttime) {
                            if (E_spike_total_ptr[over_vth_ptr[j]] != 0) {
                                double a_private = 0;
                                for (int l = 0; l<nInput_; ++l) {
                                    a_private += In_E_weight_ptr[l*nE_+over_vth_ptr[j]];
                                }
                                prior_ptr[over_vth_ptr[j]] = a_private;
                                adaptation_ptr[over_vth_ptr[j]] = 1;
                                post_spike.emplace_back(i);
                                for (int l = lasttime ; l >= 0 ; --l) {
                                    if (E_spike_ptr[over_vth_ptr[j]*time + l] == 1) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                    else if (l == 0) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                }
                                for (int l = 0; l < nInput_ ; ++l){
                                    int start = *(post_spike.end()-1);
                                    int end = *post_spike.begin();
                                    for (int m = start; m < end; ++m) {
                                        if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+over_vth_ptr[j]] != 0) {
                                            pre_spike.emplace_back(m);
                                        }
                                    }
                                    int* pre_spike_ptr = pre_spike.data();
                                    int pre_spike_size = (int)pre_spike.size();
                                    if (pre_spike_size != 0) {
                                        for (int m = 0; m<pre_spike_size; ++m) {
                                            In_E_weight_ptr[l*nE_+over_vth_ptr[j]] -=  0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+over_vth_ptr[j]];
                                        }
                                    }
                                    pre_spike.clear();
                                }
                                post_spike.clear();
                                if (adaptation_ptr[over_vth_ptr[j]] == 1) {
                                    double b_private = 0;
                                    for (int l = 0; l<nInput_; ++l) {
                                        b_private += In_E_weight_ptr[l*nE_+over_vth_ptr[j]];
                                    }
                                    latter_ptr[over_vth_ptr[j]] = b_private;
                                    if (latter_ptr[over_vth_ptr[j]] - prior_ptr[over_vth_ptr[j]]<0) {
                                        E_dCon_E_ptr[over_vth_ptr[j]] *= latter_ptr[over_vth_ptr[j]]/(prior_ptr[over_vth_ptr[j]]*spikes_ptr[over_vth_ptr[j]]);
                                        after_save_prt[over_vth_ptr[j]] = (1-E_dCon_E_ptr[over_vth_ptr[j]])/recover_constant;
                                    }
                                    else if (latter_ptr[over_vth_ptr[j]] - prior_ptr[over_vth_ptr[j]]>0){
                                        E_dCon_E_ptr[over_vth_ptr[j]] *= prior_ptr[over_vth_ptr[j]]/(latter_ptr[over_vth_ptr[j]]*spikes_ptr[over_vth_ptr[j]]);
                                        after_save_prt[over_vth_ptr[j]] = (1-E_dCon_E_ptr[over_vth_ptr[j]])/recover_constant;
                                    }
                                    adaptation_ptr[over_vth_ptr[j]] = 0;
                                }
                            }
                            else {
                                if(E_dCon_E_ptr[over_vth_ptr[j]] < 1) {
                                    E_dCon_E_ptr[over_vth_ptr[j]] += after_save_prt[over_vth_ptr[j]];
                                }
                            }
                        }
                        else {
                            if(E_dCon_E_ptr[over_vth_ptr[j]] < 1) {
                                E_dCon_E_ptr[over_vth_ptr[j]] += after_save_prt[over_vth_ptr[j]];
                            }
                        }
                    }
                    else {
                        if (learn_stable_ptr[trained_neuron] < 6) {
                            E_spike_ptr[trained_neuron*time + i] = 1;
                            E_spike_total_ptr[trained_neuron] +=1;
                            E_potential_ptr[trained_neuron] = v_reset_E;
                            time_check[trained_neuron].clear();
                            if (input_data4[trained_neuron] != 0) {
                                input_data_copy_ptr[i*nE_+trained_neuron] = 1;
                            }
                            verifying_E_ptr[trained_neuron] = 1;
                            //#pragma omp critical
                            {
                                for (int m = 0; m<nE_; ++m) {
                                    if (m != trained_neuron) {
                                        double weight = I_E_weight_prt[m*nE_+trained_neuron];
                                        /*if (neuron_exclusion == nE_) {
                                         if (neuron_index_ptr[m] != neuron_index_ptr[trained_neuron]) {
                                         weight *= weight_strength;
                                         }
                                         }*/
                                        I_data_ptr[i*nE_+m] += weight;
                                    }
                                }
                            }
                            double a_private = 0;
                            for (int l = 0; l<nInput_; ++l) {
                                a_private += In_E_weight_ptr[l*nE_+trained_neuron];
                            }
                            prior_ptr[trained_neuron] = a_private;
                            adaptation_ptr[trained_neuron] = 1;
                            post_spike.emplace_back(i);
                            for (int l = i-1 ; l >= 0 ; --l) {
                                if (E_spike_ptr[trained_neuron*time + l] == 1) {
                                    post_spike.emplace_back(l);
                                    break;
                                }
                                else if (l == 0) {
                                    post_spike.emplace_back(l);
                                    break;
                                }
                            }
                            for (int l = 0; l < nInput_ ; ++l){
                                int start = *(post_spike.end()-1);
                                int end = *post_spike.begin();
                                
                                for (int m = start; m < end; ++m) {
                                    if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+trained_neuron] != 0) {
                                        pre_spike.emplace_back(m);
                                    }
                                }
                                int pre_spike_size = (int)pre_spike.size();
                                int* pre_spike_ptr = pre_spike.data();
                                if (pre_spike_size != 0) {
                                    for (int m = 0; m<pre_spike_size; ++m) {
                                        if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == 1){
                                            In_E_weight_ptr[l*nE_+trained_neuron] += 0.01 * exp(-1*(double)(end - pre_spike_ptr[m])/addition)* (1-In_E_weight_ptr[l*nE_+trained_neuron]);
                                        }
                                        else if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == -1){
                                            In_E_weight_ptr[l*nE_+trained_neuron] -= 0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+trained_neuron];
                                        }
                                    }
                                }
                                pre_spike.clear();
                            }
                            post_spike.clear();
                            if (adaptation_ptr[trained_neuron] == 1) {
                                double b_private = 0;
                                for (int l = 0; l<nInput_; ++l) {
                                    b_private += In_E_weight_ptr[l*nE_+trained_neuron];
                                }
                                latter_ptr[trained_neuron] = b_private;
                                if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]<0) {
                                    E_dCon_E_ptr[trained_neuron] *= latter_ptr[trained_neuron]/(prior_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                    after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                }
                                else if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]>0){
                                    E_dCon_E_ptr[trained_neuron] *= prior_ptr[trained_neuron]/(latter_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                    after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                }
                                adaptation_ptr[trained_neuron] = 0;
                                spikes_ptr[trained_neuron] *= 1.022;
                            }
                        }
                        else {
                            if (neuron_index_ptr[trained_neuron] == train_label) {
                                E_spike_ptr[trained_neuron*time + i] = 1;
                                E_spike_total_ptr[trained_neuron] +=1;
                                E_potential_ptr[trained_neuron] = v_reset_E;
                                time_check[trained_neuron].clear();
                                if (input_data4[trained_neuron] != 0) {
                                    input_data_copy_ptr[i*nE_+trained_neuron] = 1;
                                }
                                verifying_E_ptr[trained_neuron] = 1;
                                //#pragma omp critical
                                {
                                    for (int m = 0; m<nE_; ++m) {
                                        if (m != trained_neuron) {
                                            double weight = I_E_weight_prt[m*nE_+trained_neuron];
                                            /*if (neuron_exclusion == nE_) {
                                             if (neuron_index_ptr[m] != neuron_index_ptr[trained_neuron]) {
                                             weight *= weight_strength;
                                             }
                                             }*/
                                            I_data_ptr[i*nE_+m] += weight;
                                        }
                                    }
                                }
                                double a_private = 0;
                                for (int l = 0; l<nInput_; ++l) {
                                    a_private += In_E_weight_ptr[l*nE_+trained_neuron];
                                }
                                prior_ptr[trained_neuron] = a_private;
                                adaptation_ptr[trained_neuron] = 1;
                                post_spike.emplace_back(i);
                                for (int l = i-1 ; l >= 0 ; --l) {
                                    if (E_spike_ptr[trained_neuron*time + l] == 1) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                    else if (l == 0) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                }
                                for (int l = 0; l < nInput_ ; ++l){
                                    int start = *(post_spike.end()-1);
                                    int end = *post_spike.begin();
                                    for (int m = start; m < end; ++m) {
                                        if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+trained_neuron] != 0) {
                                            pre_spike.emplace_back(m);
                                        }
                                    }
                                    int pre_spike_size = (int)pre_spike.size();
                                    int* pre_spike_ptr = pre_spike.data();
                                    if (pre_spike_size != 0) {
                                        for (int m = 0; m<pre_spike_size; ++m) {
                                            if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == 1){
                                                In_E_weight_ptr[l*nE_+trained_neuron] += 0.01 * exp(-1*(double)(end - pre_spike_ptr[m])/addition)* (1-In_E_weight_ptr[l*nE_+trained_neuron]);
                                            }
                                            else if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == -1){
                                                In_E_weight_ptr[l*nE_+trained_neuron] -= 0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+trained_neuron];
                                            }
                                        }
                                    }
                                    pre_spike.clear();
                                }
                                post_spike.clear();
                                if (adaptation_ptr[trained_neuron] == 1) {
                                    double b_private = 0;
                                    for (int l = 0; l<nInput_; ++l) {
                                        b_private += In_E_weight_ptr[l*nE_+trained_neuron];
                                    }
                                    latter_ptr[trained_neuron] = b_private;
                                    if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]<0) {
                                        E_dCon_E_ptr[trained_neuron] *= latter_ptr[trained_neuron]/(prior_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                        after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                    }
                                    else if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]>0){
                                        E_dCon_E_ptr[trained_neuron] *= prior_ptr[trained_neuron]/(latter_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                        after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                    }
                                    adaptation_ptr[trained_neuron] = 0;
                                    spikes_ptr[trained_neuron] *= 1.022;
                                }
                            }
                            else {
                                int* train_assign_ptr = train_assign.data();
                                int train_assing_size = (int)train_assign.size();
                                E_spike_ptr[trained_neuron*time + i] = 1;
                                E_spike_total_ptr[trained_neuron] +=1;
                                E_potential_ptr[trained_neuron] = v_reset_E;
                                time_check[trained_neuron].clear();
                                if (input_data4[trained_neuron] != 0) {
                                    input_data_copy_ptr[i*nE_+trained_neuron] = 1;
                                }
                                verifying_E_ptr[trained_neuron] = 1;
                                //#pragma omp critical
                                {
                                    for (int m = 0; m<nE_; ++m) {
                                        if (m != trained_neuron) {
                                            double weight = I_E_weight_prt[m*nE_+trained_neuron];
                                            /*if (neuron_exclusion == nE_) {
                                             if (neuron_index_ptr[m] != neuron_index_ptr[trained_neuron]) {
                                             weight *= weight_strength;
                                             }
                                             }*/
                                            I_data_ptr[i*nE_+m] += weight;
                                        }
                                    }
                                }
                                double a_private = 0;
                                for (int l = 0; l<nInput_; ++l) {
                                    a_private += In_E_weight_ptr[l*nE_+trained_neuron];
                                }
                                prior_ptr[trained_neuron] = a_private;
                                adaptation_ptr[trained_neuron] = 1;
                                post_spike.emplace_back(i);
                                for (int l = i-1 ; l >= 0 ; --l) {
                                    if (E_spike_ptr[trained_neuron*time + l] == 1) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                    else if (l == 0) {
                                        post_spike.emplace_back(l);
                                        break;
                                    }
                                }
                                int h = 0;
                                for (int l = 0; l < nInput_ ; ++l){
                                    if (train_assign_ptr[h] == l) {
                                        int start = *(post_spike.end()-1);
                                        int end = *post_spike.begin();
                                        for (int m = start; m < end; ++m) {
                                            if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+trained_neuron] != 0) {
                                                pre_spike.emplace_back(m);
                                            }
                                        }
                                        int pre_spike_size = (int)pre_spike.size();
                                        int* pre_spike_ptr = pre_spike.data();
                                        if (pre_spike_size != 0) {
                                            for (int m = 0; m<pre_spike_size; ++m) {
                                                if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == 1){
                                                    In_E_weight_ptr[l*nE_+trained_neuron] += 0.01 * exp(-1*(double)(end - pre_spike_ptr[m])/addition)* (1-In_E_weight_ptr[l*nE_+trained_neuron]);
                                                }
                                                else if (input_data_copy_ptr[pre_spike_ptr[m]*nE_+trained_neuron] == -1){
                                                    In_E_weight_ptr[l*nE_+trained_neuron] -= 0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+trained_neuron];
                                                }
                                            }
                                        }
                                        pre_spike.clear();
                                        if (h < train_assing_size-1) {
                                            h += 1;
                                        }
                                    }
                                    else {
                                        int start = *(post_spike.end()-1);
                                        int end = *post_spike.begin();
                                        for (int m = start; m < end; ++m) {
                                            if (train_data_temp_ptr[m*nInput_+l] == 1 && input_data_copy_ptr[m*nE_+over_vth_ptr[j]] != 0) {
                                                pre_spike.emplace_back(m);
                                            }
                                        }
                                        int* pre_spike_ptr = pre_spike.data();
                                        int pre_spike_size = (int)pre_spike.size();
                                        if (pre_spike_size != 0) {
                                            for (int m = 0; m<pre_spike_size; ++m) {
                                                In_E_weight_ptr[l*nE_+over_vth_ptr[j]] -=  0.0001 * exp((double)(start - pre_spike_ptr[m])/subtraction)*In_E_weight_ptr[l*nE_+over_vth_ptr[j]];
                                            }
                                        }
                                        pre_spike.clear();
                                    }
                                }
                                post_spike.clear();
                                if (adaptation_ptr[trained_neuron] == 1) {
                                    double b_private = 0;
                                    for (int l = 0; l<nInput_; ++l) {
                                        b_private += In_E_weight_ptr[l*nE_+trained_neuron];
                                    }
                                    latter_ptr[trained_neuron] = b_private;
                                    if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]<0) {
                                        E_dCon_E_ptr[trained_neuron] *= latter_ptr[trained_neuron]/(prior_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                        after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                    }
                                    else if (latter_ptr[trained_neuron] - prior_ptr[trained_neuron]>0){
                                        E_dCon_E_ptr[trained_neuron] *= prior_ptr[trained_neuron]/(latter_ptr[trained_neuron]*spikes_ptr[trained_neuron]);
                                        after_save_prt[trained_neuron] = (1-E_dCon_E_ptr[trained_neuron])/recover_constant;
                                    }
                                    adaptation_ptr[trained_neuron] = 0;
                                    spikes_ptr[trained_neuron] *= 1.022;
                                }
                            }
                        }
                    }
                }
                /*for (int j = 0; j<nE_; ++j) {
                    if (E_spike_total_ptr[j] != 0) {
                        cout << i <<"neuron: "<< j << ' ' << E_potential_ptr[j] << ' ';
                    }
                    else {
                        cout << E_potential_ptr[j] << ' ';
                    }
                }
                cout << '\n';*/
                over_vth.clear();
            }
            weight_check_In_E = 0;
        }
    }
    for (int i = 0; i<nE_; i+=4) {
        E_total_spike += vaddvq_s32(vld1q_s32(&E_spike_total_ptr[i]));
    }
}


            
void SNN::process_data(int label, int &performance_count, int iter, int total_data, int epoch, int train_gap){
    int nE_ = this->nE_;
    float temp = 0;
    vector<int> num_assignment(10,0);
    int* num_assignment_ptr = num_assignment.data();
    vector<float> summed_rate(10,0);
    float* summed_rate_ptr = summed_rate.data();
    int iter1 = iter % train_gap;
    index_num_ptr[iter1] = label;
//#pragma omp parallel
    {
//#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if(E_spike_total_ptr[i]!= 0){
                    neuron_index_num_ptr[i*train_gap+iter1] = E_spike_total_ptr[i];
                }
                for (int j = 0; j < 10; ++j) {

                        if (neuron_index_ptr[i] == j) {
//#pragma omp critical
                            {
                                num_assignment_ptr[j] += 1;
                            }
                                break;
                    }
                }
            }
        }
//#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (E_spike_total_ptr[i]!= 0) {
                    for (int j = 0; j < 10; ++j) {
                        if (neuron_index_ptr[i] == j) {
                            summed_rate_ptr[j] += (float)E_spike_total_ptr[i]/(float)num_assignment_ptr[j];
                            break;
                            
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < 10; ++j) {
        if (summed_rate_ptr[j] > temp) {
            temp = summed_rate_ptr[j];
        }
    }
    for (int j = 0; j < 10; ++j) {
        if (summed_rate_ptr[j] == temp) {
            temp = (float)j;
            break;
        }
    }
    if (label == (int)temp) {
        performance_count += 1;
    }
}

void SNN::set_index(int train_gap, int train_step, int iteration){
    vector<float>temp(10,0);
    float* temp_ptr = temp.data();
    vector<float>neuron_total_spikes(nE_*10,0);
    float* neuron_total_spikes_ptr = neuron_total_spikes.data();
    vector<int>neuron_index_copy(nE_,0);
    int* neuron_index_copy_ptr = neuron_index_copy.data();
    neuron_exclusion = 0;
    float total = 0;
    float mean = 0;
    float vari = 0;
    float stan_dev = 0;
    int32x4_t zero4 = vdupq_n_s32(0);
    for (int i = 0; i < nE_; ++i) {
        neuron_index_copy_ptr[i] = neuron_index_ptr[i];
        for (int j = 0; j<train_gap; ++j) {
            if (neuron_index_num_ptr[i*train_gap+j] != 0) {
                for (int k = 0; k<10; ++k) {
                    if (index_num_ptr[j]==k) {
                        neuron_total_spikes_ptr[i*10+k] += (float)neuron_index_num_ptr[i*train_gap+j];
                        break;
                    }
                }
            }
        }
    }
    for (int j = 0; j<train_gap; ++j) {
        for (int k = 0; k<10; ++k) {
            if (index_num_ptr[j]==k) {
                temp_ptr[k] +=1;
                break;
            }
        }
    }
    
    //#pragma omp single nowait
    {
        for (int i = 0; i < nE_; ++i) {
            for (int k = 0; k<10; ++k) {
                total += neuron_total_spikes_ptr[i*10+k];
            }
        }
        mean = total/nE_;
        double neuron_spikes = 0;
        for (int i = 0; i < nE_; ++i){
            neuron_spikes = 0;
            for (int k = 0; k<10; ++k) {
                neuron_spikes += neuron_total_spikes_ptr[i*10+k];
            }
            vari += pow(neuron_spikes-mean,2);
        }
        vari = vari / nE_;
        stan_dev = sqrt(vari);
        cout << "stan_dev: " << stan_dev << '\n';
        for (int i = 0; i < nE_; ++i) {
            float f = 0;
            for (int k = 0; k<10; ++k) {
                f += neuron_total_spikes_ptr[i*10+k];
            }
            cout << f << ' ';
            if(f==0){
                E_Con_E_ptr[i]=1;
            }
            else {
                neuron_exclusion += 1;
            }
            if (i % 80 == 79) {
                cout << '\n';
            }
        }
    }
    
    for (int i = 0; i < nE_; ++i) {
        for (int j=0 ; j<10; j+=2) {
            float32x2_t spike4_N = vld1_f32(&neuron_total_spikes_ptr[i*10+j]);
            float32x2_t spike4 = vld1_f32(&temp_ptr[j]);
            float32x2_t rate = vdiv_f32(spike4_N, spike4);
            vst1_f32(&neuron_total_spikes_ptr[i*10+j], rate);
        }
    }
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                float big = 0;
                for (int j=0 ; j<10; ++j) {
                    if (neuron_total_spikes_ptr[i*10+j] != 0) {
                        if (neuron_total_spikes_ptr[i*10+j] > big) {
                            big = neuron_total_spikes_ptr[i*10+j];
                        }
                    }
                }
                if (big != 0) {
                    for (int j=0 ; j<10; ++j) {
                        if (neuron_total_spikes_ptr[i*10+j] == big) {
                            neuron_index_ptr[i] = j;
                            break;
                        }
                    }
                    if (neuron_index_ptr[i] != neuron_index_copy_ptr[i] && iteration > 9999 ) {
                        learn_stable_ptr[i] = 1;
                    }
                    else if(iteration > train_gap - 1) {
                        learn_stable_ptr[i] += 1;
                    }
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_*train_gap; i+=4) {
                vst1q_s32(&(neuron_index_num_ptr[i]), zero4);
            }
        }
    }
}

/*void SNN::STDP(int time){
    int nE_ = this.nE_;
    int nInput_ = this.nInput_;
    float64x2_t temp2 = vdupq_n_f64(0.0f);
#pragma omp parallel
    {
        vector<int> post_spike;
        vector<int> pre_spike;
        vector<int>::iterator iter;
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (E_spike_total_ptr[i] != 0) {
                    if (learn_stable [(i) == 1) {
                        learn [(i) *= rate_dev [(i);
                    }
                    else {
                        learn [(i) *= pow(rate_dev [(i), learn_stable . at(i));
                    }
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE_; ++i) {
                if (E_spike_total_ptr[i] != 0) {
                    for (int j = 0 ; j < time ; ++j) {
                        if (j == 0) {
                            post_spike.emplace_back(j);
                        }
                        else if (E_spike_ptr[i*time+j) == 1) {
                            post_spike.emplace_back(j);
                        }
                    }
                    if (post_spike.size() > 1) {
                        for (int k = 0; k < nInput_ ; ++k){
                            for (iter = post_spike.begin(); iter != post_spike.end(); ++iter) {
                                if (iter != (post_spike.end()-1)) {
                                    if (iter == post_spike.begin()) {
                                        if (*(iter+1) < time - 5) {
                                            for (int m = *iter; m<*(iter+1)+5; ++m) {
                                                if (train_data_temp[(m*nInput_+k) == 1) {
                                                    pre_spike.emplace_back(m);
                                                }
                                            }
                                        }
                                        else {
                                            for (int m = *iter; m<time; ++m) {
                                                if (train_data_temp[(m*nInput_+k) == 1) {
                                                    pre_spike.emplace_back(m);
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (*(iter+1) < time - 5) {
                                            for (int m = *iter+5 ; m<*(iter+1)+5; ++m) {
                                                if (train_data_temp[(m*nInput_+k) == 1) {
                                                    pre_spike.emplace_back(m);
                                                }
                                            }
                                        }
                                        else {
                                            for (int m = *iter+5; m<time; ++m) {
                                                if (train_data_temp[(m*nInput_+k) == 1) {
                                                    pre_spike.emplace_back(m);
                                                }
                                            }
                                        }
                                    }
                                    if (pre_spike.size() != 0) {
                                        for (int l = 0; l<pre_spike.size(); ++l) {
                                            if (*(iter+1) - pre_spike_ptr[m] >= 0) {
                                                if (input_data_copy[(pre_spike_ptr[m]*nE_+i) == 1){
                                                    In_E_weight_ptr[k*nE_+i) += 0.01 * exp(-1*(double)(*(iter+1) - pre_spike_ptr[m])/20)* pow(1-In_E_weight_ptr[k*nE_+i), 0.9);
                                                }
                                            }
                                            else {
                                                double mm = 0.0001 * exp((double)(*(iter+1) - pre_spike_ptr[m])/40)*pow(In_E_weight_ptr[k*nE_+i),0.9);
                                                if (In_E_weight_ptr[k*nE_+i) < mm) {
                                                    In_E_weight_ptr[k*nE_+i) = 0;
                                                }
                                                else {
                                                    In_E_weight_ptr[k*nE_+i) -= mm;
                                                }
                                            }
                                        }
                                    }
                                }
                                else if(iter == (post_spike.end()-1) && *(post_spike.end()-1) < time - 6) {
                                    for (int m = *iter+5; m<time; ++m) {
                                        if (train_data_temp[(m*nInput_+k) == 1) {
                                            pre_spike.emplace_back(m);
                                        }
                                    }
                                    if (pre_spike.size() != 0) {
                                        for (int l = 0; l<pre_spike.size(); ++l) {
                                            double mm = 0.0001 * exp((double)(*iter - pre_spike_ptr[m])/40)*pow(In_E_weight_ptr[k*nE_+i),0.9);
                                            if (In_E_weight_ptr[k*nE_+i) < mm) {
                                                In_E_weight_ptr[k*nE_+i) = 0;
                                            }
                                            else {
                                                In_E_weight_ptr[k*nE_+i) -= mm;
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
#pragma omp for
    {
        for (int i = 0; i < nE_*time; i+=2) {
            vst1q_f64(&input_data_copy[(i), temp2);
        }
    }
}*/

void SNN::resting(int time, vector<double> &after_save){
#pragma omp parallel
    {
        double gE_constant = (1 - time_step_ / tau_syn_E);
        double gI_constant = (1 - time_step_ / tau_syn_I);
        double E_time_con = time_step_/tau_E;
        int nE_ = this->nE_;
        vector<double> I_data_sum(nE_, 0);
        float64x2_t gL4 = vdupq_n_f64(gL);
        float64x2_t v_reset_E4 = vdupq_n_f64(v_reset_E);
#pragma omp for
            {
                for (int j = 0; j < nE_; j+=2){
                    for (int i = 0; i < time; ++i) {
                        float64x2_t E_rest = vdupq_n_f64(v_rest_E);
                        float64x2_t E_vE_E = vdupq_n_f64(vE_E);
                        float64x2_t E_vI_E = vdupq_n_f64(vI_E);
                        
                        float64x2_t E_Potential4 = vld1q_f64(&(E_potential_ptr[j]));
                        
                        /* Sensory_gE = Sensory_gE * (1 - time_step_ / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max */
                        float64x2_t E_Con_E4 = vld1q_f64(&(E_Con_E_ptr[j]));
                        if (vaddvq_f64(E_Con_E4) != 0) {
                            E_Con_E4 = vmulq_n_f64(E_Con_E4, gE_constant);
                            vst1q_f64(&(E_Con_E_ptr[j]), E_Con_E4);
                            E_Con_E4 = vdivq_f64(E_Con_E4, gL4);
                        }
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /* Sensory_gI = Sensory_gI * (1 - time_step_ / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max */
                        float64x2_t E_Con_I4 = vld1q_f64(&(E_Con_I_ptr[j]));
                        if (vaddvq_f64(E_Con_I4) != 0) {
                            E_Con_I4 = vmulq_n_f64(E_Con_I4, gI_constant);
                            vst1q_f64(&(E_Con_I_ptr[j]), E_Con_I4);
                            E_Con_I4 = vdivq_f64(E_Con_I4, gL4);
                        }
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        /*  Sensory_dv = (-(Sensory_potential - v_rest_e) - (Sensory_gE/gL)*(Sensory_potential-vE_E)- (Sensory_gI/gL) * (Sensory_potential - vI_E))*(time_step_ /tau_e) */
                        E_rest = vsubq_f64(E_rest, E_Potential4);
                        E_vE_E = vsubq_f64(E_vE_E, E_Potential4);
                        E_vI_E = vsubq_f64(E_vI_E, E_Potential4);
                        
                        float64x2_t E_dPotential = vmlaq_f64(E_rest, E_Con_E4, E_vE_E);
                        E_dPotential = vmlaq_f64(E_dPotential, E_Con_I4, E_vI_E);
                        
                        E_dPotential = vmulq_n_f64(E_dPotential, E_time_con);
                        E_Potential4 = vaddq_f64(E_Potential4, E_dPotential);
                        E_Potential4 = vmaxq_f64(v_reset_E4, E_Potential4);
                        vst1q_f64(&(E_potential_ptr[j]), E_Potential4);
                        /* end */
                    }
                }
            /* Inhibitory */
/*#pragma omp for
            {
                for (int j = 0; j < nE_; j+=2){
                    float64x2_t I_rest = vdupq_n_f64(v_rest_I);
                    float64x2_t I_vE_I = vdupq_n_f64(vE_I);
                    
                    if (i<3) {
                        if (i == 0) {
                            verifying_I = {vaddv_f64(vld1_f64(&I_spike_previous[(j*(time+1) + (time+1) - 3))) + I_spike_previous[(j*(time+1) + (time+1)), vaddv_f64(vld1_f64(&I_spike_previous[((j+1)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+1)*(time+1) + (time+1) - 1), vaddv_f64(vld1_f64(&I_spike_previous[((j+2)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+2)*(time+1) + (time+1) - 1), vaddv_f64(vld1_f64(&I_spike_previous[((j+3)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+3)*(time+1) + (time+1) - 1)};
                        }
                        else {
                            verifying_I = {vaddv_f64(vld1_f64(&I_spike_previous[(j*(time+1) + (time+1) - 3))) + I_spike_previous[(j*(time+1) + (time+1)), vaddv_f64(vld1_f64(&I_spike_previous[((j+1)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+1)*(time+1) + (time+1) - 1), vaddv_f64(vld1_f64(&I_spike_previous[((j+2)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+2)*(time+1) + (time+1) - 1), vaddv_f64(vld1_f64(&I_spike_previous[((j+3)*(time+1) + (time+1) - 3))) + I_spike_previous[((j+3)*(time+1) + (time+1) - 1)};
                            for (int k = 3; k > 3-i; --k) {
                                verifying_I[0] = verifying_I[0] - I_spike_previous[(j*(time+1) + (time+1) - k) + I_spike_. at(j*(time+1) +  3 - k);
                                verifying_I[1] = verifying_I[1] - I_spike_previous[((j+1)*(time+1) + (time+1) - k) + I_spike_. at((j+1)*(time+1) + 3 - k);
                                verifying_I[2] = verifying_I[2] - I_spike_previous[((j+2)*(time+1) + (time+1) - k) + I_spike_. at((j+2)*(time+1) + 3 - k);
                                verifying_I[3] = verifying_I[3] - I_spike_previous[((j+3)*(time+1) + (time+1) - k) + I_spike_. at((j+3)*(time+1) + 3 - k);
                            }
                        }
                        float64x2_t I_Potential4 = vld1q_f64(&(I_potential_[(j)));
                        
                         //E_to_I_spike_data = weight_S_E * Sensory_spike[:,i]
                        // Inter_I_gE = Inter_I_gE * (1 - time_step_ / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
                        float64x2_t E_spike4 = {E_spike_ptr[j*time + i), E_spike_ptr[(j+1)*time + i), E_spike_ptr[(j+2)*time + i), E_spike_ptr[(j+3)*time + i)};
                        float64x2_t I_Con_E4 = vld1q_f64(&(I_Con_E_[(j)));
                        I_Con_E4 = vmulq_n_f64(I_Con_E4, gE_constant);
                        if (vaddvq_f64(E_spike4) != 0.0f) {
                            float64x2_t weight_E_I4 = vld1q_f64(&(E_I_weight_[(j)));
                            float64x2_t E_I_spike_data = vmulq_f64(weight_E_I4, E_spike4);
                            float64x2_t I_dCon_E4 = vld1q_f64(&(I_dCon_E_[(j)));
                            E_I_spike_data = vmulq_f64(E_I_spike_data, I_dCon_E4);
                            I_Con_E4 = vaddq_f64(I_Con_E4, E_I_spike_data);
                        }
                        vst1q_f64(&(I_Con_E_[(j)), I_Con_E4);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        
                        //  Inter_dv_I = (-(Inter_potential_I - v_rest_i) - (Inter_I_gE/gL)*(Inter_potential_I-vE_I))*(time_step_ /tau_i)
                        
                        I_rest = vsubq_f64(I_rest, I_Potential4);
                        
                        
                        I_vE_I = vsubq_f64(I_vE_I, I_Potential4);
                        I_Con_E4 = vdivq_f64(I_Con_E4, gL4);
                        
                        float64x2_t I_dPotential = vmlaq_f64(I_rest, I_Con_E4, I_vE_I);
                        
                        I_dPotential = vmulq_n_f64(I_dPotential, I_time_con);
                        
                        I_Potential4 = vaddq_f64(I_Potential4, I_dPotential);
                        for (int k = 0; k < 4; ++k) {
                            if (verifying_I[k] != 0 && I_Potential4[k] > v_reset_I) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        I_Potential4 = vmaxq_f64(v_rest_I4, I_Potential4);
                        uint32x4_t I_spike4_1 = vcgtq_f64(I_Potential4, v_thresh_I4);
                        uint32x4_t I_spike4_int = vshrq_n_u32(I_spike4_1, 31);
                        float64x2_t I_spike4 = vcvtq_f64_u32(I_spike4_int);
                        I_spike_[(j*351 + i+1) = I_spike4[0];
                        I_spike_[((j+1)*351 + i+1) = I_spike4[1];
                        I_spike_[((j+2)*351 + i+1) = I_spike4[2];
                        I_spike_[((j+3)*351 + i+1) = I_spike4[3];
                        for (int k = 0; k < 4; ++k) {
                            if (I_spike4[k] == 1) {
                                I_Potential4[k] = v_reset_I;
                            }
                        }
                        vst1q_f64(&(I_potential_[(j)), I_Potential4);
                    }
                    else {
                        float64x2_t verifying_0 = vld1q_f64(&(I_spike_[(j*time + i-3)));
                        float64x2_t verifying_1 = vld1q_f64(&(I_spike_[((j+1)*time + i-3)));
                        float64x2_t verifying_2 = vld1q_f64(&(I_spike_[((j+2)*time + i-3)));
                        float64x2_t verifying_3 = vld1q_f64(&(I_spike_[((j+3)*time + i-3)));
                        verifying_0[4] = 0;
                        verifying_1[4] = 0;
                        verifying_2[4] = 0;
                        verifying_3[4] = 0;
                        float64x2_t I_Potential4 = vld1q_f64(&(I_potential_[(j)));
                        verifying_I = {vaddvq_f64(verifying_0), vaddvq_f64(verifying_1), vaddvq_f64(verifying_2), vaddvq_f64(verifying_3)};
                        
                        // E_to_I_spike_data = weight_S_E * Sensory_spike[:,i]
                        // Inter_I_gE = Inter_I_gE * (1 - time_step_ / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
               2         }
                        vst1q_f64(&(I_potential_[(j)), I_Potential4);
                    }
                }
            }*/
        }
    }
}

void SNN::normalization(vector<double> &In_E, int nInput, int nE){
    vector<double> temp(nE,0);
    double* In_E_ptr = In_E.data();
#pragma omp parallel
    {
        float64x2_t scalar_vector = vdupq_n_f64(78);
#pragma omp for
        {
            for (int j = 0; j < nE; ++j) {
                for (int i = 0; i < nInput; i+=2) {
                    float64x2_t weight;
                    weight[0]=  In_E_ptr[i*nE + j];
                    weight[1]=  In_E_ptr[(i+1)*nE + j];
                    temp[j] += vaddvq_f64(weight);
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE; i+=2) {
                float64x2_t temp2 = vld1q_f64(&temp[i]);
                float64x2_t result = vdivq_f64(scalar_vector, temp2);
                vst1q_f64(&temp[i], result);
            }
        }
#pragma omp for
        {
            for (int j = 0; j < nE; ++j)  {
                for (int i = 0; i < nInput; i+=2) {
                    float64x2_t weight;
                    weight[0]=  In_E_ptr[i*nE + j];
                    weight[1]=  In_E_ptr[(i+1)*nE + j];
                    float64x2_t result = vmulq_n_f64(weight, temp[j]);
                    In_E_ptr[i*nE + j] =  result[0];
                    In_E_ptr[(i+1)*nE + j] =  result[1];
                }
            }
        }
    }
}

/*void SNN::spike_total(vector<double> &E_spike, vector<double> &E_spike_total, int nE, int simulate_time, double &E_total_spike){
    double E_total_spike_temp = 0.0f;
#pragma omp parallel
    {
        double E_total_spike_temp_private = 0;
#pragma omp for
        {<
            for (int i = 0; i<nE; i+=2) {
                float64x2_t total_spike = vld1q_f64(&E_spike_total_ptr[i]);
                E_total_spike_temp_private += vaddvq_f64(total_spike);
            }
        }
#pragma omp critical
            {
                E_total_spike_temp += E_total_spike_temp_private;
            }
#pragma omp barrier
    }
    E_total_spike = E_total_spike_temp;
}*/

/*void SNN::train_modulation() {
#pragma omp parallel
    {
#pragma omp for
        {
            
        }
    }
}*/

