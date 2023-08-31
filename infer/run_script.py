from run_infer_cross import run
from Lizard_eva_cross import compute_pq_mpq
from class_F1_utils.class_wise_mask import generate_class_label
from class_F1_utils.PanNuke_cross_classonly_eva import run_nuclei_type_stat
import os
import shutil
import numpy as np

refer = 'mix_np_tp3_tp6_gradient=10^3_allpseutrain_bothSVD_nointerval_merge_flat_tp_residual=0.1_bankinterval=1'
os.makedirs('./pred/Lizard/' + refer + '/', exist_ok = True)
fp_all = open('./pred/Lizard/' + refer + '/eva_mpq+.txt', "w")

i = 15
PRETRAINED = './exp_output/local/models/' + refer + '/00/model/net_epoch=' + str(i) + '.tar'
save_paths = './pred/Lizard/' + refer + '/' + str(i) + '/'
run(PRETRAINED, save_paths)

#fp = open(save_paths + 'eva_mpq+.txt', "w")
true_dir = 'Please type in the path for the test_mask file'
pred_dir = save_paths +  'valid_pred.npy'
eva_results = compute_pq_mpq(pred_dir, true_dir, 6)
#fp.writelines(pred_dir + '\n')
#fp.writelines('mpq+ score of this model is: ' + str(eva_results['multi_pq+']) + '\n')
#fp.writelines('bpq score of this model is: ' + str(eva_results['pq']) + '\n')
#fp.writelines('class-wise pq score of this model is: ' + str(eva_results['class_pq']) + '\n')
#fp.close()

few_class_mean_pq = (eva_results['class_pq'][0] + eva_results['class_pq'][3] + eva_results['class_pq'][4]) / 3
few_class_mean_dice = (eva_results['class_dice'][0] + eva_results['class_dice'][3] + eva_results['class_dice'][4]) / 3
fp_all.writelines(str(i) + '\n')
fp_all.writelines('mpq+ score of this model is: ' + str(eva_results['multi_pq+']) + '\n')
fp_all.writelines('bpq score of this model is: ' + str(eva_results['pq']) + '\n')
fp_all.writelines('few_class mean pq score of this model is: ' + str(few_class_mean_pq) + '\n')
fp_all.writelines('class-wise pq score of this model is: ' + str(eva_results['class_pq']) + '\n')
fp_all.writelines('mdice+ score of this model is: ' + str(eva_results['multi_dice+']) + '\n')
fp_all.writelines('bdice score of this model is: ' + str(eva_results['dice']) + '\n')
fp_all.writelines('few_class mean dice score of this model is: ' + str(few_class_mean_dice) + '\n')
fp_all.writelines('class-wise dice score of this model is: ' + str(eva_results['class_dice']) + '\n')

#####################################################################################################
generate_class_label(save_paths)

true_class_dir = 'Please type in the path towards the class label folder for the test set'
pred_class_dir = save_paths + 'pred_label_class/'
class_result = run_nuclei_type_stat(pred_class_dir, true_class_dir)
few_class_mean_f1 = (class_result[2] + class_result[5] + class_result[6]) / 3
fp_all.writelines('class-wise f1 score of this model is: ' + str(class_result[2:]) + '\n')
fp_all.writelines('mean f1 score of this model is: ' + str(np.mean(class_result[2:])) + '\n')
fp_all.writelines('few_class mean f1 score of this model is: ' + str(few_class_mean_f1) + '\n')


#os.remove(save_paths + 'valid_pred.npy')
#shutil.rmtree(save_paths + 'pred_label_class/')
#shutil.rmtree(save_paths + 'subfile/')
#shutil.rmtree(save_paths + 'vis/')
shutil.rmtree(save_paths)
    
fp_all.close()





