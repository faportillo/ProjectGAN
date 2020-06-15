# ProjectGAN
Using different Generative Adversarial Networks (GANs) to generate images.

*Currently supported models: 

 - DCGAN
	 - [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf) 
![Image Completion with Deep Learning in TensorFlow](https://bamos.github.io/data/2016-08-09/discrim-architecture.png)

 - SAGAN
	 - [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)![GAN — Self-Attention Generative Adversarial Networks (SAGAN)](https://miro.medium.com/max/3200/1*oIAw_f4Zw6iJfFU6TbeoaA.jpeg)

Currently supported datasets: 

 - CelebA


Each model was trained on a single NVIDIA K40 GPU for 10 epochs. 
Uses the ADAM Optimizer with different learning rates for both the Discriminator and Generator. 
______
**Results**
Results taken from the output of the Generator and compared to the "Real Images" from the CelebA dataset.

DCGAN:
![DCGAN](https://drive.google.com/uc?export=view&id=1KvhH0ve72PNrFWdtyimqnErKsbIPfOlo)

SAGAN:
![SAGAN](https://drive.google.com/uc?export=view&id=1zSKToizjKIFKgz9C6eg6WGs4LsrqN5RG)

 
