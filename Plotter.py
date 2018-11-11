import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

name1='WGANLosses'
train_log_from_file1=np.loadtxt(name1)
train_log_from_file1=np.array([train_log_from_file1[i][1] for i in range(len(train_log_from_file1))])
print(train_log_from_file1[0])
ax=sns.tsplot(data=train_log_from_file1, color='green', condition='Wasserstein GAN', legend=True)
print(train_log_from_file1.shape)

'''
name2='GANLosses'
train_log_from_file2=np.loadtxt(name2)
print(train_log_from_file2.shape)
ax2=sns.tsplot(data=train_log_from_file2, color='red', condition='Vanilla GAN', legend=True)


name3='HalfCheetah_HybridPPO_lr0.0001_sigma0.05.txt'
train_log_from_file3=np.loadtxt(name3)
print(train_log_from_file3.shape)
ax3=sns.tsplot(data=train_log_from_file3, color='blue', condition='lr=.0001', legend=True)

name4='HalfCheetah_HybridPPO_lr0.0005_sigma0.05.txt'
train_log_from_file4=np.loadtxt(name4)
print(train_log_from_file4.shape)
ax4=sns.tsplot(data=train_log_from_file4, color='yellow', condition='lr=.0005', legend=True)


name5='HalfCheetah_Vanilla_sigma0.05_lr1e-05_SR0.2_gamma0.99_EL200trials5_batch64_lam0.95.txt'
train_log_from_file5=np.loadtxt(name5)
print(train_log_from_file5.shape)
ax5=sns.tsplot(data=train_log_from_file5, color='red', condition='Vanilla GA sigma=.05', legend=True)
'''

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
print('gonna show')
plt.show()

