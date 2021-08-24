/*
 * @Author: your name
 * @Date: 2021-08-05 11:50:43
 * @LastEditTime: 2021-08-05 13:41:22
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\error_code.h
 */
// Define error code and message for keyword spotting which can be a common tools
#ifndef ERROR_CODE_H
#define ERROR_CODE_H

#include <stdexcept>
#include <iostream>
#include <unordered_map>

typedef enum ErrorCode {
	// Rules to define error code name, E_axx_bxx_cxx, part-a means module, part-b means function, 
	// part-c means specific error, eg: e_Mfcc_param_setting, parameters setting error in mfcc module
	// 0 success, 1 common error
	// 50~99, errors appears in audio process
	// 100~199 errors appears in mfcc extraction module
	// 200~299 errors appears in Keyword Spotting(prediction) module
    I_success = 0,		// success
	E_common = 1,

	// 50 ~
	E_wav_open = 50,				// load wav file error
	E_wav_open_outfile,				// open output file error
	E_wav_open_folder,				// open wav folder error
	E_wav_audio_format,					// audio format was not 1 or 16 bit PCM wave or 1-channel

	// 100 ~
	E_mfcc_param_setting = 100,		// code 100, mfcc parameter setting error
	E_mfcc_param_rate,				// sample rate error was not 16000
	E_mfcc_param_window_size,		// window size was not 30(ms)
	E_mfcc_param_window_stride,		// window stride was not 10
	E_mfcc_param_clip_duration,		// audio clip duration was not 1000(ms)
	E_mfcc_param_upper_frequency,	// upper frequency was not 4000hz
	E_mfcc_param_lower_frequency,	// lower frequency was not 20hz
	E_mfcc_param_filterbank_channel,// filter bank channel was not 40
	E_mfcc_param_dct_coefficient,	// dct coefficient was not 40
	E_mfcc_param_feature_length,	// feature length was not 3920
	E_mfcc_param_not_exist,			// parameter was not exist
	E_mfcc_param_spectrogram_init,	// Spectrogram initialize error


	// 200 ~
	E_tflite_create_interpreter = 200,				// tflite interpreter create not succeed
	E_tflite_allocate_tensor,				// tflite interpreter allocate tensor error
	E_tflite_input_tensor_name,				// tflite get input tensor name error
	E_tflite_output_tensor_name,			// tfliate get output tensor name error
	E_tflite_label_name,					// label number not the same
	E_tflite_logit_label_num,				// logit size was not the same as label number

}; // error code define



// Define error code to messages
extern std::unordered_map<int, std::string> err_code_to_msg;

//std::unordered_map<int, std::string> err_code_to_msg = {
//	{I_success, "Succeed"},
//	{E_common , "Common error"},
//	{E_wav_open, "load wav file error"},
//	{E_wav_open_outfile, "open output file error"},
//	{E_wav_open_folder,	"open wav folder error"},
//	{E_wav_audio_format, "audio format was not 1 or 16 bit PCM wave or 1-channel"},
//
//	{E_mfcc_param_setting,	"mfcc parameter setting error"},
//	{E_mfcc_param_rate,	"sample rate error was not 16000"},
//	{E_mfcc_param_window_size, "window size was not 30(ms)"},
//	{E_mfcc_param_window_stride,"window stride was not 10"	},
//	{E_mfcc_param_clip_duration,"audio clip duration was not 1000(ms)"}, 
//	{E_mfcc_param_upper_frequency, "upper frequency was not 4000hz"	},
//	{E_mfcc_param_lower_frequency,	"lower frequency was not 20hz"}, 
//	{E_mfcc_param_filterbank_channel, "filter bank channel was not 40"}, 
//	{E_mfcc_param_dct_coefficient, "dct coefficient was not 40"},
//	{E_mfcc_param_feature_length, "feature length was not 3920"}, 
//	{E_mfcc_param_not_exist, "parameter was not exist"}, 
//
//	{E_tflite_create_interpreter, "tflite interpreter create not error"},
//	{E_tflite_allocate_tensor, "tflite interpreter allocate tensor error"}, 
//	{E_tflite_input_tensor_name, "tflite get input tensor name error"}, 
//	{E_tflite_output_tensor_name, "tflite get output tensor name error"}, 
//	{E_tflite_label_name, "label number not the same"},
//	{E_tflite_logit_label_num, "logit size was not the same as label number"}	
//};


#endif // error_code.h