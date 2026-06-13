import torch
from typing import List

import numpy as np

from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
import os

class HybridSafetyProjectorV3:
    """
    双模态混合安全子空间投影器
    使用广义特征分解构建视觉和文本两个维度的安全投影子空间
    视觉子空间：Σ_mal_vis u = λ Σ_ben_vis u
    文本子空间：Σ_mal_text v = λ Σ_ben_text v
    """
    
    def __init__(
        self,
        benign_acts_visual: dict,
        malicious_acts_visual: dict,
        benign_acts_text: dict,
        malicious_acts_text: dict,
        target_layers: list = [9, 10],
        n_components_visual: int = 64,  # 视觉恶意方向数
        n_components_text: int = 64,    # 文本恶意方向数
        projection_strength_visual: float = 0.6,
        projection_strength_text: float = 0.6,
        fusion_mode: str = 'sequential',  # 'sequential', 'parallel', 'adaptive'
        visual_text_ratio: float = 0.5,   # 视觉和文本权重比例（parallel模式）
        model_type: str = "qwen"
    ):
        """
        Args:
            benign_acts_visual: {layer_idx: np.array [N, hidden_dim]} - 视觉良性激活
            malicious_acts_visual: {layer_idx: np.array [N, hidden_dim]} - 视觉恶意激活
            benign_acts_text: {layer_idx: np.array [N, hidden_dim]} - 文本良性激活
            malicious_acts_text: {layer_idx: np.array [N, hidden_dim]} - 文本恶意激活
            n_components_visual: 视觉子空间维度
            n_components_text: 文本子空间维度
            projection_strength_visual: 视觉投影强度
            projection_strength_text: 文本投影强度
            fusion_mode: 
                - 'sequential': 先视觉后文本（级联）
                - 'parallel': 视觉和文本并行加权
                - 'adaptive': 自适应选择stronger方向
            visual_text_ratio: parallel模式下的权重比例 (0-1, 0.5表示等权重)
        """
        self.target_layers = target_layers
        self.n_components_visual = n_components_visual
        self.n_components_text = n_components_text
        self.projection_strength_visual = projection_strength_visual
        self.projection_strength_text = projection_strength_text
        self.fusion_mode = fusion_mode
        self.visual_text_ratio = visual_text_ratio
        
        # 视觉子空间
        self.projectors_visual = {}  # U_k_vis ∈ R^{d x k_vis}
        self.benign_centers_visual = {}
        self.benign_means_visual = {}
        
        # 文本子空间
        self.projectors_text = {}  # U_k_text ∈ R^{d x k_text}
        self.benign_centers_text = {}
        self.benign_means_text = {}
        
        # 转向向量（结合视觉和文本信息）
        self.steering_vectors = {}
        
        print("="*80)
        print(f"初始化双模态混合安全投影器 (Visual + Text Generalized Subspace)")
        print(f"融合模式: {fusion_mode.upper()}")
        print(f"视觉子空间: {n_components_visual}维, 强度: {projection_strength_visual}")
        print(f"文本子空间: {n_components_text}维, 强度: {projection_strength_text}")
        if fusion_mode == 'parallel':
            print(f"Visual:Text权重比 = {visual_text_ratio:.2f}:{1-visual_text_ratio:.2f}")
        print("="*80)
        
        for layer in target_layers:
            benign_vis = benign_acts_visual[layer]
            malicious_vis = malicious_acts_visual[layer]
            benign_txt = benign_acts_text[layer]
            malicious_txt = malicious_acts_text[layer]
            
            print(f"\n{'='*60}")
            print(f"Layer {layer}:")
            print(f"  Visual  - Benign: {benign_vis.shape}, Malicious: {malicious_vis.shape}")
            print(f"  Text    - Benign: {benign_txt.shape}, Malicious: {malicious_txt.shape}")

            # ========== 第一部分：视觉安全子空间 ==========
            cache_dir = f"./cache/{model_type}/v_subspace"
            os.makedirs(cache_dir, exist_ok=True)
            cache_prefix = f"{cache_dir}/visual_subspace_layer{layer}_comp{n_components_visual}"
            cache_U = f"{cache_prefix}_U.pt"
            cache_mean = f"{cache_prefix}_mean.pt"
            cache_center = f"{cache_prefix}_center.pt"
            print(f"\n  [Visual Subspace]")
            if os.path.exists(cache_U) and os.path.exists(cache_mean) and os.path.exists(cache_center):
                print(f"  [Cache Found] Loading from {cache_prefix}_*.pt")
                U_k_vis = torch.load(cache_U, map_location="cpu").numpy()
                benign_mean_vis = torch.load(cache_mean, map_location="cpu").numpy()
                benign_center_vis = torch.load(cache_center, map_location="cpu").numpy()
            else:
                U_k_vis, benign_mean_vis, benign_center_vis = self._build_generalized_subspace(
                benign_vis, malicious_vis, 
                n_components=n_components_visual,
                subspace_name="Visual"
            )
                torch.save(torch.tensor(U_k_vis), cache_U)
                torch.save(torch.tensor(benign_mean_vis), cache_mean)
                torch.save(torch.tensor(benign_center_vis), cache_center)
                print(f"  [Cache Saved] -> {cache_prefix}_*.pt")
            
            self.projectors_visual[layer] = torch.tensor(U_k_vis, dtype=torch.float16)
            self.benign_centers_visual[layer] = torch.tensor(benign_center_vis, dtype=torch.float16)
            self.benign_means_visual[layer] = torch.tensor(benign_mean_vis, dtype=torch.float16)
            
            # ========== 第二部分：文本安全子空间 ==========
            cache_dir = f"./cache/{model_type}/t_subspace"
            os.makedirs(cache_dir, exist_ok=True)
            cache_prefix = f"{cache_dir}/text_subspace_layer{layer}_comp{n_components_text}"
            cache_U = f"{cache_prefix}_U.pt"
            cache_mean = f"{cache_prefix}_mean.pt"
            cache_center = f"{cache_prefix}_center.pt"
            print(f"\n  [Text Subspace]")
            if os.path.exists(cache_U) and os.path.exists(cache_mean) and os.path.exists(cache_center):
                print(f"  [Cache Found] Loading from {cache_prefix}_*.pt")
                U_k_txt = torch.load(cache_U, map_location="cpu").numpy()
                benign_mean_txt = torch.load(cache_mean, map_location="cpu").numpy()
                benign_center_txt = torch.load(cache_center, map_location="cpu").numpy()
            else:
                U_k_txt, benign_mean_txt, benign_center_txt = self._build_generalized_subspace(
                    benign_txt, malicious_txt,
                    n_components=n_components_text,
                    subspace_name="Text"
                )
                torch.save(torch.tensor(U_k_txt), cache_U)
                torch.save(torch.tensor(benign_mean_txt), cache_mean)
                torch.save(torch.tensor(benign_center_txt), cache_center)
                print(f"  [Cache Saved] -> {cache_prefix}_*.pt")
            
            self.projectors_text[layer] = torch.tensor(U_k_txt, dtype=torch.float16)
            self.benign_centers_text[layer] = torch.tensor(benign_center_txt, dtype=torch.float16)
            self.benign_means_text[layer] = torch.tensor(benign_mean_txt, dtype=torch.float16)

            # ========== 分析子空间正交性 ==========
            orthogonality = self._analyze_subspace_orthogonality(U_k_vis, U_k_txt)
            print(f"\n  [Subspace Analysis]")
            print(f"    Visual-Text orthogonality: {orthogonality:.4f} (1.0=完全正交)")
        
        print("\n" + "="*80)
        print("双模态混合安全投影器初始化完成")
        print("="*80)
    
    def _build_generalized_subspace(self, benign_acts, malicious_acts, n_components, subspace_name=""):
        """
        构建广义特征分解的安全子空间
        返回: U_k, benign_mean, benign_center
        """
        benign_mean = benign_acts.mean(axis=0)
        malicious_mean = malicious_acts.mean(axis=0)
        
        # 中心化
        benign_centered = benign_acts - benign_mean
        malicious_centered = malicious_acts - malicious_mean
        
        # 计算协方差矩阵
        Sigma_ben = benign_centered.T @ benign_centered / len(benign_acts)
        Sigma_mal = malicious_centered.T @ malicious_centered / len(malicious_acts)
        
        # 正则化
        eps = 1e-6
        d = Sigma_ben.shape[0]
        Sigma_ben_reg = Sigma_ben + eps * np.eye(d)
        
        # 白化矩阵
        from scipy.linalg import sqrtm
        Sigma_ben_sqrt_inv = np.linalg.inv(sqrtm(Sigma_ben_reg))
        
        # 广义矩阵
        A = Sigma_ben_sqrt_inv @ Sigma_mal @ Sigma_ben_sqrt_inv
        
        # 求 top-k 特征向量
        k = min(n_components, A.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1][:k]
        V_k = eigvecs[:, idx]
        
        # 映射回原空间
        U_k = Sigma_ben_sqrt_inv @ V_k
        U_k = U_k.astype(np.float32)
        
        # 归一化
        for i in range(k):
            U_k[:, i] /= (np.linalg.norm(U_k[:, i]) + 1e-8)
        
        # 分析
        benign_norm = np.linalg.norm(benign_centered, axis=1).mean()
        malicious_norm = np.linalg.norm(malicious_centered, axis=1).mean()
        mal_in_U = U_k.T @ malicious_mean
        energy_ratio = np.linalg.norm(mal_in_U) / (np.linalg.norm(malicious_mean) + 1e-8)
        
        print(f"    {subspace_name} - k={k}, Benign norm: {benign_norm:.4f}, Mal norm: {malicious_norm:.4f}")
        print(f"    {subspace_name} - Mal energy in U_k: {energy_ratio:.2%}")
        
        return U_k, benign_mean, benign_mean
    
    def _compute_dual_modal_steering(self, benign_vis, mal_vis, benign_txt, mal_txt, 
                                     method='hybrid', normalize=True):
        """
        计算双模态转向向量，结合视觉和文本信息
        """
        # 视觉转向
        steering_vis = self._compute_steering_vector(benign_vis, mal_vis, method, normalize=False)
        
        # 文本转向
        steering_txt = self._compute_steering_vector(benign_txt, mal_txt, method, normalize=False)
        
        # 融合策略
        if method == 'hybrid':
            # 加权平均，文本权重稍高（因为文本更直接反映语义）
            steering_vec = 0.4 * steering_vis + 0.6 * steering_txt
        else:
            steering_vec = 0.5 * steering_vis + 0.5 * steering_txt
        
        if normalize:
            steering_vec = steering_vec / (np.linalg.norm(steering_vec) + 1e-8)
        
        return steering_vec
    
    def _analyze_subspace_orthogonality(self, U_vis, U_txt):
        """
        分析视觉和文本子空间的正交性
        返回: 平均余弦相似度（越接近0越正交）
        """
        # 计算所有列向量对之间的余弦相似度
        similarities = []
        for i in range(U_vis.shape[1]):
            for j in range(U_txt.shape[1]):
                cos_sim = np.abs(np.dot(U_vis[:, i], U_txt[:, j]))
                similarities.append(cos_sim)
        
        avg_similarity = np.mean(similarities)
        orthogonality = 1.0 - avg_similarity  # 转换为正交性指标
        return orthogonality
    
    def project_and_steer(self, activations: torch.Tensor, layer_idx: int):
        """
        双模态混合操作：结合视觉和文本子空间投影
        """
        # 获取投影矩阵
        U_vis = self.projectors_visual[layer_idx].to(activations.device)
        U_txt = self.projectors_text[layer_idx].to(activations.device)
        center_vis = self.benign_centers_visual[layer_idx].to(activations.device)
        center_txt = self.benign_centers_text[layer_idx].to(activations.device)
        benign_mean_vis = self.benign_means_visual[layer_idx].to(activations.device)
        benign_mean_txt = self.benign_means_text[layer_idx].to(activations.device)
        
        original_shape = activations.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            flat_act = activations.view(-1, original_shape[-1])
        else:
            flat_act = activations
        
        # ========== 双模态投影 ==========
        if self.fusion_mode == 'sequential':
            # 级联模式：先视觉后文本
            corrected = self._apply_subspace_projection(
                flat_act, U_vis, center_vis, benign_mean_vis, 
                self.projection_strength_visual
            )
            corrected = self._apply_subspace_projection(
                corrected, U_txt, center_txt, benign_mean_txt,
                self.projection_strength_text
            )
            
        elif self.fusion_mode == 'parallel':
            # 并行模式：视觉和文本加权融合
            corrected_vis = self._apply_subspace_projection(
                flat_act, U_vis, center_vis, benign_mean_vis,
                self.projection_strength_visual
            )
            corrected_txt = self._apply_subspace_projection(
                flat_act, U_txt, center_txt, benign_mean_txt,
                self.projection_strength_text
            )
            # 加权融合
            w_vis = self.visual_text_ratio
            w_txt = 1.0 - w_vis
            corrected = w_vis * corrected_vis + w_txt * corrected_txt
            
        elif self.fusion_mode == 'adaptive':
            # 自适应模式：选择correction更强的方向
            corrected_vis = self._apply_subspace_projection(
                flat_act, U_vis, center_vis, benign_mean_vis,
                self.projection_strength_visual
            )
            corrected_txt = self._apply_subspace_projection(
                flat_act, U_txt, center_txt, benign_mean_txt,
                self.projection_strength_text
            )
            # 计算correction强度
            delta_vis = torch.norm(corrected_vis - flat_act, dim=-1, keepdim=True)
            delta_txt = torch.norm(corrected_txt - flat_act, dim=-1, keepdim=True)
            # 自适应权重
            total = delta_vis + delta_txt + 1e-8
            w_vis = delta_vis / total
            w_txt = delta_txt / total
            corrected = w_vis * corrected_vis + w_txt * corrected_txt
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        if is_3d:
            corrected = corrected.view(original_shape)
        
        return corrected
    
    def _apply_subspace_projection(self, flat_act, U_k, center, benign_mean, strength):
        """
        应用单个子空间投影（对齐注入）
        h' = h - α * comp_mal + α * benign_mal_component
        """
        # 中心化
        centered = flat_act - center
        
        # 恶意方向投影矩阵
        proj_mal = U_k @ U_k.t()
        
        # 投影到恶意子空间
        comp_mal = torch.mm(centered, proj_mal.t())
        
        # 良性参考在恶意子空间的分量
        benign_centered = (benign_mean - center).unsqueeze(0)
        benign_mal_component = torch.mm(benign_centered, proj_mal.t())
        
        # 对齐注入
        corrected = flat_act - strength * comp_mal + strength * benign_mal_component
        
        return corrected
    
    def register_hooks(self, model):
        """
        注册前向hook到目标层的post_attention_layernorm之后
        这样可以干预residual stream而不影响attention/mlp的输入输出格式
        
        Returns:
            hooks列表，用于后续移除
        """
        hooks = []
        base_model = model.model if hasattr(model, 'model') else model.base_model
        
        print("\n注册双模态混合投影hooks（residual stream干预）...")
        for layer_idx in self.target_layers:
            layer_module = base_model.language_model.layers[layer_idx]
            
            # 尝试hook MLP的输出（最安全的位置）
            if hasattr(layer_module, 'mlp'):
                target_module = layer_module.mlp
                hook_location = "MLP"
            # 如果没有mlp，尝试hook post_attention_layernorm
            elif hasattr(layer_module, 'post_attention_layernorm'):
                target_module = layer_module.post_attention_layernorm
                hook_location = "post_attention_layernorm"
            # 最后尝试input_layernorm
            elif hasattr(layer_module, 'input_layernorm'):
                target_module = layer_module.input_layernorm
                hook_location = "input_layernorm"
            else:
                print(f"  ✗ Layer {layer_idx}: 找不到合适的hook位置")
                continue
            
            def make_hook(layer_id):
                def hook_fn(module, input, output):
                    # 直接处理输出，不需要担心tuple解包问题
                    processed = self.project_and_steer(output, layer_id)
                    return processed
                return hook_fn
            
            hook = target_module.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
            print(f"  ✓ Layer {layer_idx} {hook_location} hook registered")
        
        return hooks
