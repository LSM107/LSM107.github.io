$$
\begin{bmatrix}

5.4 & 2.21 & 1.8 & 4.2

\end{bmatrix}

\\

\times
$$

$$
J^{\pi_\theta} = E[\sum_{t=0}^\infty\gamma^tr_{t+1}|\pi_\theta, \rho] = J^{\pi_{\theta_k}}+
E[\sum_{t=0}^\infty\gamma^k A^{\pi_{\theta_k}}(s_t, a_t) |\pi_\theta, \rho]
$$




$$
E[V^{\pi_{\theta_k}}(s_0)|\rho]+
E[\sum_{t=0}^\infty\gamma^t (\gamma V^{\pi_{\theta_k}}(s_{t+1}) + r_{t+1} - V^{\pi_{\theta_k}}(s_t)) |\pi_\theta, \rho]
$$

$$
J^{\pi_{\theta_k}}+
E[\sum_{t=0}^\infty\gamma^t (\gamma V^{\pi_{\theta_k}}(s_{t+1}) + r_t - V^{\pi_{\theta_k}}(s_t)) |\pi_\theta, \rho]
$$

$$
J^{\pi_{\theta_k}}+
E[\sum_{t=0}^\infty\gamma^t E[(\gamma V^{\pi_{\theta_k}}(s_{t+1}) + r_t - V^{\pi_{\theta_k}}(s_t))|s_t, a_t] |\pi_\theta, \rho]
$$

- Iterative expectation (not rigorous math expressions)




$$
J^{\pi_\theta} = 
J^{\pi_{\theta_k}}+
E[\sum_{t=0}^\infty\gamma^k A^{\pi_{\theta_k}}(s_t, a_t) |\pi_\theta, \rho] = 
J^{\pi_{\theta_k}}+
\sum_{t=0}^{\tau - 1}\gamma^t A^{\pi_{\theta_k}}(s_t, a_t)
$$

- Stochastic approximation, Episode should be generated with $\pi_\theta$	
  - Hard to take gradient in $\theta$




$$
l^{CLIP}(\theta) = \sum_{t=0}^{\tau-1} \gamma^t\min[
\hat{A}^{\pi_{\theta_k}}(s_t, a_t)\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}, 
\hat{A}^{\pi_{\theta_k}}(s_t, a_t)clip[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}, 1-\epsilon, 1+\epsilon]
]
$$

$$
= \sum_{t=0}^{\tau-1} \gamma^t\min[
\hat{A}^{\pi_{\theta_k}}(s_t, a_t)\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}, 
\hat{A}^{\pi_{\theta_k}}(s_t, a_t)(1-\epsilon)]
$$

$$
= \sum_{t=0}^{\tau-1}
\gamma^t\hat{A}^{\pi_{\theta_k}}(s_t, a_t)(1-\epsilon)
$$



