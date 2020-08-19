from tsnecuda import TSNE
import numpy as np
model = '200819_BiC_CIFAR10_bic_0_memsz_2000_base_5_step_5_batch_128_epoch_250_distill_kd_task_5'
data = np.load('./feature_result/'+model+'.npy')

task = 0
X = data[task].reshape(-1, 64)
print(X.shape, data.shape)
X_embedded = TSNE(n_components=2).fit_transform(X)

print(X_embedded.shape)
np.save('./feature_result/'+model+'_tsne.npy', X_embedded)
