  Wozu indices? (Was beinhaltet memory?) attppo L79
  Was ist rollout in ML?, Why shuffle? attppo L207/9
  Was ist der syntax in ppo L490?

  Wie ist GAE implementiert? gae
Forward gae over trajectories stored in memory?!

  Wie ist attention sinnvoll?
SaledDPAttention(K, V, Q) = softmax(QK^T(sqrt(d_k)))V
(k, v, q) = x_i * (K, V, Q)

q -> Statespezifischer Abfragevektor
k -> Statespezifische Features die abgefragt werden können
q * k -> Abfragen der Features
v -> Skalierung für q * k
