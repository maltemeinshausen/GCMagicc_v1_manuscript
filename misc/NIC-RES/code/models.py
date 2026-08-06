import torch.nn as nn
import healpy as hp
import numpy as np
import torch
import torch.nn.functional as F
import torch.sparse as sparse
from scipy.spatial import cKDTree

def make_mlp(in_dim, out_dim, hidden_dim, nlayers, addtanh = False):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ELU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    if addtanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


from help_functions import Downsample
    

class L4adde(nn.Module):
    
    def __init__(
        self,
        nside: int,
        *,
        n_features: int = 4,
        x_features: int = 10,
        x_dim: int = 4,
        map_dim: int = 15,
        noise_dim_aug: int = 5,
        noise_dim_high: int = 5,
        num_neighbors: int = 16,
        mlp_hidden: int = 32,
        mlp_depth: int = 4,
        noise_x: int = 4,
        variab: int = 0,
        num_max: int = 35,
        addconstant = False
    ):
        super().__init__()
        # ─── scaling configuration -----------------------------------
        self.var_dims = max(0, int(variab))            # #encoded dims to scale
        self.num_max  = int(num_max)

        # ─── geometry -------------------------------------------------
        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        # ─── feature / noise dims ------------------------------------
        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high
        self.lags           = None                     # no temporal dependence
        self.addconstant = addconstant
        
        # ─── neighbour indices (hi-res) ------------------------------
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T
        idx = cKDTree(vec).query(vec, k=self.k)[1]
        self.register_buffer("neighbor_indices",
                             torch.tensor(idx.clip(0, self.npix - 1),
                                          dtype=torch.long))          # (P,k)

        # ─── learnable sparse-map weights ---------------------------
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        n_features + noise_dim_aug)
        )
        nn.init.xavier_uniform_(self.weight_map)
        
        self.weight_map2 = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        map_dim)
        )
        nn.init.xavier_uniform_(self.weight_map2)

        
        if self.noise_dim_high:
            self.weight_map_noisehigh = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh)
            self.weight_map_noisehigh2 = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh2)
            
        # ─── class-specific scale vectors ---------------------------
        if self.var_dims > 0:
            # (num_max+1, var_dims)  – exp() ⇒ positive
            self.scale_raw = nn.Parameter(torch.zeros(self.num_max + 1,
                                                      self.var_dims))
        else:
            self.register_parameter("scale_raw", None)

        # ─── global-x encoder --------------------------------------
        self.noise_x = noise_x
                

        # ─── residual MLP ------------------------------------------
        self.mlp_in_dim = map_dim + noise_dim_high 
        self.latent_mlp = make_mlp(self.mlp_in_dim,
                                   n_features,
                                   mlp_hidden,
                                   mlp_depth)

    # -----------------------------------------------------------------
    def _forward_single(self, x, y_in):
        """
        x    : (B, x_features)   – full input vector
        y_in : (B, n_features, P)
        """
        B, dev = x.size(0), x.device

        # ===== sparse map ============================================
        noise_aug = torch.randn(B, self.noise_dim_aug, self.npix, device=dev)
        if self.addconstant:
            with torch.no_grad():
                noise_aug[:,0,:] = 1.0
                    
        y_cat     = torch.cat([y_in, noise_aug], dim=1)                  # (B,C,P)
        gather    = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :] # (B,P,k,C)
        sparse    = torch.einsum("bpkn,pkmn->bpm", gather,
                                 self.weight_map)                       # (B,P,map)
        sparse    = torch.einsum("bpkn,pkmn->bpm", sparse[:, self.neighbor_indices, :],
                                 self.weight_map2)  
        # ===== encode x (drop first column) ==========================
        
        # ===== noise for MLP =========================================
        noise_high = (torch.randn(B, self.npix, self.noise_dim_high, device=dev)) if self.noise_dim_high else None
        if self.addconstant:
            with torch.no_grad():
                noise_high[..., 0] = 1.0
                    
        noise_high = torch.einsum("bpkn,pkmn->bpm",
                                noise_high[:, self.neighbor_indices, :] , self.weight_map_noisehigh)
        noise_high = torch.einsum("bpkn,pkmn->bpm",
                                noise_high[:, self.neighbor_indices, :] , self.weight_map_noisehigh2)
            
        
        # ===== residual prediction ===================================
        parts = [sparse]
        if noise_high is not None:
            parts.append(noise_high)
        mlp_in  = torch.cat(parts, dim=2)                                   # (B,P,D)
        residual = self.latent_mlp(mlp_in).permute(0, 2, 1)                 # (B,F,P)
        return y_in + residual

    # -----------------------------------------------------------------
    def forward(self, x, y_in, *, dependence: bool = False, **ignored):
        if dependence:
            raise ValueError("L2f has no temporal dependence")

        orig_2d = x.dim() == 2
        if orig_2d:
            x, y_in = x.unsqueeze(0), y_in.unsqueeze(0)

        outs = [self._forward_single(x[t], y_in[t]) for t in range(x.size(0))]
        out  = torch.stack(outs, dim=0)                                     # (T,B,F,P)
        return out.squeeze(0) if orig_2d else out

class L4addemini(nn.Module):
    def __init__(
        self,
        nside: int,
        *,
        n_features: int = 4,
        x_features: int = 10,
        x_dim: int = 4,
        map_dim: int = 10,
        noise_dim_aug: int = 1,
        noise_dim_high: int = 1,
        num_neighbors: int = 6,
        mlp_hidden: int = 16,
        mlp_depth: int = 3,
        addconstant: bool = False,
    ):
        super().__init__()

        # ─── geometry ────────────────────────────────────────────────
        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        # ─── feature / noise dims ────────────────────────────────────
        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high
        self.lags           = None
        self.addconstant    = addconstant

        # ─── neighbour indices (hi-res) ─────────────────────────────
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T  # (P,3)
        idx = cKDTree(vec).query(vec, k=self.k)[1]                  # (P,k)
        self.register_buffer(
            "neighbor_indices",
            torch.tensor(idx.clip(0, self.npix - 1), dtype=torch.long),
            persistent=False,
        )

        # ─── learnable sparse-map weights ───────────────────────────
        # weights for y_in + low-noise: (P, k, map_dim, C)
        C = n_features + noise_dim_aug
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim, C)
        )
        nn.init.xavier_uniform_(self.weight_map)

        # weights for high-noise mixing: (P, k, H, H)
        H = self.noise_dim_high
        if H > 0:
            self.weight_map_noisehigh = nn.Parameter(
                torch.empty(self.npix, self.k, H, H)
            )
            nn.init.xavier_uniform_(self.weight_map_noisehigh)
        else:
            self.weight_map_noisehigh = None

        # ─── residual MLP over [map_dim (+ H if used)] ──────────────
        self.mlp_in_dim = map_dim + (noise_dim_high if noise_dim_high else 0)
        self.latent_mlp = make_mlp(self.mlp_in_dim, n_features, mlp_hidden, mlp_depth)

    # -----------------------------------------------------------------
    def _forward_single(self, x: torch.Tensor, y_in: torch.Tensor, indices: None) -> torch.Tensor:
        
        B, nF, P = y_in.shape
        dev   = y_in.device
        dtype = y_in.dtype
        H     = self.noise_dim_high

        # ---- build inputs once (cheap), then operate chunked over P ----
        noise_aug = torch.randn(B, self.noise_dim_aug, P, device=dev, dtype=dtype)
        if self.addconstant and self.noise_dim_aug > 0:
            with torch.no_grad():
                noise_aug[:, 0, :] = 1.0
        y_in_aug = torch.cat([y_in, noise_aug], dim=1).permute(0, 2, 1)    # (B, C, P)

        # high-noise (per target pixel) built for all P, then chunked
        if H and H > 0:
            noise_high_full = torch.randn(B, P, H, device=dev, dtype=dtype)
            if self.addconstant:
                with torch.no_grad():
                    noise_high_full[..., 0] = 1.0
        else:
            noise_high_full = None

        out = torch.empty(B, nF, P if indices is None else len(indices), device=dev, dtype=dtype)
        
        if indices is None:
            idx = self.neighbor_indices        # (Pc, k)
        else:
            idx = self.neighbor_indices[indices]         # (Pc, k)

            
        # gather: (B, Pc, k, C)
        gather = y_in_aug[:, idx, :]
        # weights: (Pc, k, map_dim, C)
        if indices is None:
            w_map = self.weight_map
        else:
            w_map = self.weight_map[indices]
            
        #w_map = F.softmax(w_map, dim=-1)
        # sparse: (B, Pc, map_dim)
        sparse = torch.einsum("bpkn,pkmn->bpm", gather, w_map)

        # ===== high noise path (optional) =========================
        if noise_high_full is not None:
            # (B, Pc, k, H)
            nh_gather = noise_high_full[:, idx, :]
            # (Pc, k, H, H)
            if indices is None:
                w_nh = self.weight_map_noisehigh
            else:   
                w_nh = self.weight_map_noisehigh[indices]
                
            # mixed high-noise: (B, Pc, H)
            noise_high_mixed = torch.einsum("bpkn,pkmn->bpm", nh_gather, w_nh)
            parts = [sparse, noise_high_mixed]
        else:
            parts = [sparse]

        # ===== residual head ======================================
        mlp_in = torch.cat(parts, dim=2)          # (B, Pc, map_dim + H?)
        residual = self.latent_mlp(mlp_in).permute(0, 2, 1)  # (B, F, Pc)

        # skip add on the same Pc
        if indices is None:
            out = y_in + residual
        else:
            out = y_in[:, :, indices] + residual


        return out  # (B, F, P)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, y_in: torch.Tensor, indices=None, *, dependence: bool = False, **_ignored) -> torch.Tensor:
        if dependence:
            raise ValueError("L4addemini has no temporal dependence")

        # Accept (B, x_features) or (T, B, x_features)
        orig_2d = x.dim() == 2
        if orig_2d:
            x = x.unsqueeze(0)     # (1, B, x_features)
            y_in = y_in.unsqueeze(0)
            
        outs = []
        for t in range(x.size(0)):
            outs.append(self._forward_single(x[t], y_in[t], indices = indices))

        out = torch.stack(outs, dim=0)  # (T, B, F, P)
        return out.squeeze(0) if orig_2d else out

class L4de(nn.Module):
    
    # -----------------------------------------------------------------
    def __init__(
        self,
        nside: int,
        *,
        # ---------- original L2f arguments ----------------------------
        n_features: int = 4,
        x_features: int = 10,
        x_dim: int = 20,
        map_dim: int = 20,
        noise_dim_aug: int = 5,
        noise_dim_high: int = 5,
        num_neighbors: int = 16,
        mlp_hidden: int = 32,
        mlp_depth: int = 4,
        depth_mlp_x: int = 4,
        hidden_x: int = 32,
        noise_x: int = 5,
        variab: int = 0,
        num_max: int = 35,
        # ---------- NEW (to match L2) ---------------------------------
        add_latent_dim: int = 8,
        noise_dim_prev: int = 5,
        lags: list[int] = (1, 2),
        addconstant = False
    ):
        super().__init__()

        # ====== keep all old config unchanged =========================
        self.var_dims = max(0, min(int(x_dim),int(variab)))
        self.num_max  = int(num_max)

        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high
        self.addconstant = addconstant 
        
        # ===== neighbour indices (high-res) ===========================
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T
        idx = cKDTree(vec).query(vec, k=self.k)[1]
        self.register_buffer("neighbor_indices",
                             torch.tensor(idx.clip(0, self.npix-1),
                                          dtype=torch.long))          # (P,k)

        # ===== learnable sparse-map weights ==========================
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        n_features + noise_dim_aug))
        nn.init.xavier_uniform_(self.weight_map)
        
        self.weight_map2 = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        map_dim))
        nn.init.xavier_uniform_(self.weight_map2)

        self.weight_map_noise = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise)
        
        
        self.weight_map_noise2 = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise2)
        
        if self.noise_dim_high:
            self.weight_map_noisehigh = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh)
            
            self.weight_map_noisehigh2 = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh2)
            
        # ===== optional class scaling =================================
        if self.var_dims > 0:
            self.scale_raw = nn.Parameter(
                torch.zeros(self.num_max+1, self.var_dims))
        else:
            self.register_parameter("scale_raw", None)

        # ===== global-x encoder =======================================
        self.noise_x = noise_x
        self.mlp_x_reduce = make_mlp(
            (x_features-1) + noise_x, x_dim, hidden_x, depth_mlp_x)

        
        
        self.lags   = list(lags)
        self.n_prev = len(self.lags)

        self.add_latent_dim = (n_features if add_latent_dim is None
                               else add_latent_dim)
        self.noise_dim_prev = (n_features if noise_dim_prev is None
                               else noise_dim_prev)

        self.D_prev = (x_dim
                       + n_features                # current y_in average
                       + self.noise_dim_prev
                       + self.n_prev * n_features)


        self.mlp_in_dim = x_dim + map_dim + noise_dim_high  + self.add_latent_dim
        self.final_mlp = make_mlp(self.mlp_in_dim,
                                   n_features,
                                   mlp_hidden,
                                   mlp_depth)
        self.latent_mlp = make_mlp(self.D_prev,
                                   self.add_latent_dim,
                                   mlp_hidden,
                                   mlp_depth)

    
    # -----------------------------------------------------------------
    # forward for a single step  (unchanged maths + extra latent)
    # -----------------------------------------------------------------
    def _forward_single(self, x, y_in, *, prev_list=None):
        """
        x      : (B, x_features)
        y_in   : (B, n_features, P)     – low-res input for this step
        prev_list : list[Tensor|None] of length n_prev,    each (B,F,P)
        """
        B, dev = x.size(0), x.device
        prev_list = prev_list or [None] * self.n_prev

        # ===== sparse-map from y_in ==================================
        noise_aug = torch.randn(B, self.noise_dim_aug,
                                self.npix, device=dev)
        if self.addconstant:
            with torch.no_grad():
                noise_aug[:,0,:] = 1.0
        y_cat   = torch.cat([y_in, noise_aug], 1)              # (B,C,P)
        ## weights (P,k,map,C)
        gather  = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :]   # (B,P,k,C)
        sparse  = torch.einsum("bpkn,pkmn->bpm",  gather, self.weight_map)        # (B,P,map)
        sparse = torch.einsum("bpkn,pkmn->bpm",   sparse[:,self.neighbor_indices,:], self.weight_map2)
        
        # ===== encode x (drop first col) =============================
        noise_vec = torch.randn(B, self.noise_x, device=dev)
        x_red = self.mlp_x_reduce(torch.cat([x[:, 1:], noise_vec], 1))   # (B,x_dim)
        x_exp = x_red.unsqueeze(1).expand(B, self.npix, self.x_dim)      # (B,P,x_dim)

        # ===== optional class-vector scaling =========================
        if self.var_dims > 0:
            cls   = x[:, 0].long().clamp_(0, self.num_max)
            scale = torch.exp(self.scale_raw[cls])                       # (B,var)
            #d     = min(self.var_dims, self.x_dim)
            d     = self.var_dims
            mask  = torch.ones_like(x_exp)
            mask[:, :, :d] = scale.view(B, 1, d)
            x_exp = x_exp * mask

        # ===== high-level noise vector ===============================
        noise_high = None
        if self.noise_dim_high:
            noise_high = (torch.randn(B, self.npix, self.noise_dim_high,
                                       device=dev))                # (B,P,noise)
            if self.addconstant:
                with torch.no_grad():
                    noise_high[..., 0] = 1.0
            noise_high = torch.einsum("bpkn,pkmn->bpm",
                                      noise_high[:, self.neighbor_indices, :] , self.weight_map_noisehigh)
            noise_high = torch.einsum("bpkn,pkmn->bpm",   noise_high[:,self.neighbor_indices,:], self.weight_map_noisehigh2)

        # ===== additional latent branch (lags) =======================
        add_latent = (torch.randn(B, self.npix,
                                self.add_latent_dim, device=dev))
        if self.addconstant:
            with torch.no_grad():
                add_latent[:, :, 0] = 1.0
                    
        add_latent = torch.einsum("bpkn,pkmn->bpm",
                                add_latent[:, self.neighbor_indices, :] , self.weight_map_noise)
        add_latent = torch.einsum("bpkn,pkmn->bpm",
                                add_latent[:, self.neighbor_indices, :] , self.weight_map_noise2)
        # (B,P,add_latent)
        
        if not any(p is None for p in prev_list) and torch.rand(1) < 0.995:

            noise_prev = torch.randn(B, self.npix,
                                        self.noise_dim_prev, device=dev)
                    
            y_naive = y_in.permute(0, 2, 1)

            prev_concat = torch.cat([p.permute(0, 2, 1) - y_naive
                                     for p in prev_list], 2)   # (B,P,F*n_prev)
            
            add_in = torch.cat([x_exp, y_naive, noise_prev, prev_concat], 2)  # (B,P,D_prev)
            
            add_latent = self.latent_mlp(add_in)                # (B,P,add_latent)
            #mask = torch.rand_like(tmp, dtype=torch.float32) < 0.95
            #add_latent = torch.where(mask, tmp, add_latent)
            
        # ===== residual prediction ===================================
        parts = [x_exp, sparse, add_latent]
        if noise_high is not None:
            parts.append(noise_high)

        mlp_in  = torch.cat(parts, 2)                         # (B,P,Dtot)
        return  self.final_mlp(mlp_in).permute(0, 2, 1)   # (B,F,P)

    # -----------------------------------------------------------------
    # public forward (single step or autoregressive sequence)
    # -----------------------------------------------------------------
    def forward(
        self,
        x,              # (..., x_features)
        y_in,           # (..., n_features, P)
        *,
        dependence: bool = False,
        prev_tensors: dict[int, torch.Tensor] | None = None,
    ):
        # --- single step --------------------------------------------
        if not dependence:
            prev_list = None
            if prev_tensors is not None:
                prev_list = [prev_tensors.get(l) for l in self.lags]
            return self._forward_single(x, y_in, prev_list=prev_list)
       

        # --- autoregressive sequence --------------------------------
        orig_2d = x.dim() == 2
        if orig_2d:
            x, y_in = x.unsqueeze(1), y_in.unsqueeze(1)       # add time dim

        T = x.size(0)
        outs = []
        for t in range(T):
            prev_list = [outs[t - l] if t >= l else None for l in self.lags]
            outs.append(self._forward_single(x[t], y_in[t],
                                             prev_list=prev_list))

        out = torch.stack(outs, 0)                            # (T,B,F,P)
        return out.squeeze(1) if orig_2d else out
    

class L4df(nn.Module):
    
    # -----------------------------------------------------------------
    def __init__(
        self,
        nside: int,
        *,
        # ---------- original L2f arguments ----------------------------
        n_features: int = 4,
        x_features: int = 10,
        x_dim: int = 20,
        map_dim: int = 20,
        noise_dim_aug: int = 5,
        noise_dim_high: int = 5,
        num_neighbors: int = 16,
        mlp_hidden: int = 32,
        mlp_depth: int = 4,
        depth_mlp_x: int = 4,
        hidden_x: int = 32,
        noise_x: int = 5,
        variab: int = 0,
        num_max: int = 35,
        # ---------- NEW (to match L2) ---------------------------------
        add_latent_dim: int = 8,
        noise_dim_prev: int = 5,
        lags: list[int] = (1, 2),
        addconstant = False
    ):
        super().__init__()

        # ====== keep all old config unchanged =========================
        self.var_dims = max(0, min(int(x_dim),int(variab)))
        self.num_max  = int(num_max)

        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high
        self.addconstant = addconstant 
        
        # ===== neighbour indices (high-res) ===========================
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T
        idx = cKDTree(vec).query(vec, k=self.k)[1]
        self.register_buffer("neighbor_indices",
                             torch.tensor(idx.clip(0, self.npix-1),
                                          dtype=torch.long))          # (P,k)

        # ===== learnable sparse-map weights ==========================
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        n_features + noise_dim_aug))
        nn.init.xavier_uniform_(self.weight_map)
        
        self.weight_map2 = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        map_dim))
        nn.init.xavier_uniform_(self.weight_map2)

        self.weight_map_noise = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise)
        
        
        self.weight_map_noise2 = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise2)
        
        if self.noise_dim_high:
            self.weight_map_noisehigh = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh)
            
            self.weight_map_noisehigh2 = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh2)
            
        # ===== optional class scaling =================================
        if self.var_dims > 0:
            self.scale_raw = nn.Parameter(
                torch.zeros(self.num_max+1, self.var_dims))
        else:
            self.register_parameter("scale_raw", None)

        # ===== global-x encoder =======================================
        self.noise_x = noise_x
        self.mlp_x_reduce = make_mlp(
            (x_features) + noise_x, x_dim, hidden_x, depth_mlp_x)

        
        
        self.lags   = list(lags)
        self.n_prev = len(self.lags)

        self.add_latent_dim = (n_features if add_latent_dim is None
                               else add_latent_dim)
        self.noise_dim_prev = (n_features if noise_dim_prev is None
                               else noise_dim_prev)

        self.D_prev = (x_dim
                       + n_features                # current y_in average
                       + self.noise_dim_prev
                       + self.n_prev * n_features)


        self.mlp_in_dim = x_dim + map_dim + noise_dim_high  + self.add_latent_dim
        self.final_mlp = make_mlp(self.mlp_in_dim,
                                   n_features,
                                   mlp_hidden,
                                   mlp_depth)
        self.latent_mlp = make_mlp(self.D_prev,
                                   self.add_latent_dim,
                                   mlp_hidden,
                                   mlp_depth, addtanh=True)

    
    # -----------------------------------------------------------------
    # forward for a single step  (unchanged maths + extra latent)
    # -----------------------------------------------------------------
    def _forward_single(self, x, y_in, *, prev_list=None):
        """
        x      : (B, x_features)
        y_in   : (B, n_features, P)     – low-res input for this step
        prev_list : list[Tensor|None] of length n_prev,    each (B,F,P)
        """
        B, dev = x.size(0), x.device
        prev_list = prev_list or [None] * self.n_prev

        # ===== sparse-map from y_in ==================================
        noise_aug = torch.randn(B, self.noise_dim_aug,
                                self.npix, device=dev)
        if self.addconstant:
            with torch.no_grad():
                noise_aug[:,0,:] = 1.0
        y_cat   = torch.cat([y_in, noise_aug], 1)              # (B,C,P)
        ## weights (P,k,map,C)
        gather  = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :]   # (B,P,k,C)
        sparse  = torch.einsum("bpkn,pkmn->bpm",  gather, self.weight_map)        # (B,P,map)
        sparse = torch.einsum("bpkn,pkmn->bpm",   sparse[:,self.neighbor_indices,:], self.weight_map2)
        
        # ===== encode x (drop first col) =============================
        noise_vec = torch.randn(B, self.noise_x, device=dev)
        x_red = self.mlp_x_reduce(torch.cat([x, noise_vec], 1))   # (B,x_dim)
        x_exp = x_red.unsqueeze(1).expand(B, self.npix, self.x_dim)      # (B,P,x_dim)

        # ===== optional class-vector scaling =========================
        if self.var_dims > 0:
            cls   = x[:, 0].long().clamp_(0, self.num_max)
            scale = torch.exp(self.scale_raw[cls])                       # (B,var)
            #d     = min(self.var_dims, self.x_dim)
            d     = self.var_dims
            mask  = torch.ones_like(x_exp)
            mask[:, :, :d] = scale.view(B, 1, d)
            x_exp = x_exp * mask

        # ===== high-level noise vector ===============================
        noise_high = None
        if self.noise_dim_high:
            noise_high = (torch.randn(B, self.npix, self.noise_dim_high,
                                       device=dev))                # (B,P,noise)
            if self.addconstant:
                with torch.no_grad():
                    noise_high[..., 0] = 1.0
            noise_high = torch.einsum("bpkn,pkmn->bpm",
                                      noise_high[:, self.neighbor_indices, :] , self.weight_map_noisehigh)
            noise_high = torch.einsum("bpkn,pkmn->bpm",   noise_high[:,self.neighbor_indices,:], self.weight_map_noisehigh2)

        # ===== additional latent branch (lags) =======================
        add_latent = (torch.randn(B, self.npix,
                                self.add_latent_dim, device=dev))
        if self.addconstant:
            with torch.no_grad():
                add_latent[:, :, 0] = 1.0
                    
        add_latent = torch.einsum("bpkn,pkmn->bpm",
                                add_latent[:, self.neighbor_indices, :] , self.weight_map_noise)
        add_latent = torch.einsum("bpkn,pkmn->bpm",
                                add_latent[:, self.neighbor_indices, :] , self.weight_map_noise2)
        # (B,P,add_latent)
        add_latent = F.tanh(add_latent)
        
        if not any(p is None for p in prev_list) and torch.rand(1) < 0.995:

            noise_prev = torch.randn(B, self.npix,
                                        self.noise_dim_prev, device=dev)
                    
            y_naive = y_in.permute(0, 2, 1)

            prev_concat = torch.cat([p.permute(0, 2, 1) - y_naive
                                     for p in prev_list], 2)   # (B,P,F*n_prev)
            
            add_in = torch.cat([x_exp, y_naive, noise_prev, prev_concat], 2)  # (B,P,D_prev)
            
            add_latent = self.latent_mlp(add_in)                # (B,P,add_latent)
            #mask = torch.rand_like(tmp, dtype=torch.float32) < 0.95
            #add_latent = torch.where(mask, tmp, add_latent)
            
        # ===== residual prediction ===================================
        parts = [x_exp, sparse, add_latent]
        if noise_high is not None:
            parts.append(noise_high)

        mlp_in  = torch.cat(parts, 2)                         # (B,P,Dtot)
        return  self.final_mlp(mlp_in).permute(0, 2, 1)   # (B,F,P)

    # -----------------------------------------------------------------
    # public forward (single step or autoregressive sequence)
    # -----------------------------------------------------------------
    def forward(
        self,
        x,              # (..., x_features)
        y_in,           # (..., n_features, P)
        *,
        dependence: bool = False,
        prev_tensors: dict[int, torch.Tensor] | None = None,
    ):
        # --- single step --------------------------------------------
        if not dependence:
            prev_list = None
            if prev_tensors is not None:
                prev_list = [prev_tensors.get(l) for l in self.lags]
            return self._forward_single(x, y_in, prev_list=prev_list)
       

        # --- autoregressive sequence --------------------------------
        orig_2d = x.dim() == 2
        if orig_2d:
            x, y_in = x.unsqueeze(1), y_in.unsqueeze(1)       # add time dim

        T = x.size(0)
        outs = []
        for t in range(T):
            prev_list = [outs[t - l] if t >= l else None for l in self.lags]
            outs.append(self._forward_single(x[t], y_in[t],
                                             prev_list=prev_list))

        out = torch.stack(outs, 0)                            # (T,B,F,P)
        return out.squeeze(1) if orig_2d else out
   
class L4des(nn.Module):
    
    # -----------------------------------------------------------------
    def __init__(
        self,
        nside: int,
        *,
        # ---------- original L2f arguments ----------------------------
        n_features: int = 10,
        x_features: int = 10,
        x_dim: int = 20,
        map_dim: int = 20,
        noise_dim_aug: int = 5,
        noise_dim_high: int = 5,
        num_neighbors: int = 16,
        mlp_hidden: int = 32,
        mlp_depth: int = 4,
        depth_mlp_x: int = 4,
        hidden_x: int = 32,
        noise_x: int = 5,
        variab: int = 0,
        num_max: int = 35,
        # ---------- NEW (to match L2) ---------------------------------
        add_latent_dim: int = 20,
        keep_prev_feat = 3,
        noise_dim_prev: int = 5,
        lags: list[int] = (1, 2),
        addconstant = False
    ):
        super().__init__()

        # ====== keep all old config unchanged =========================
        self.var_dims = max(0, min(int(x_dim),int(variab)))
        self.num_max  = int(num_max)

        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high
        self.addconstant = addconstant 
        self.keep_prev_feat = keep_prev_feat
        # ===== neighbour indices (high-res) ===========================
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T
        idx = cKDTree(vec).query(vec, k=self.k)[1]
        self.register_buffer("neighbor_indices",
                             torch.tensor(idx.clip(0, self.npix-1),
                                          dtype=torch.long))          # (P,k)

        # ===== learnable sparse-map weights ==========================
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        n_features + noise_dim_aug))
        nn.init.xavier_uniform_(self.weight_map)
        
        self.weight_map2 = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        map_dim))
        nn.init.xavier_uniform_(self.weight_map2)

        self.weight_map_noise = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise)
        
        
        self.weight_map_noise2 = nn.Parameter(
            torch.empty(self.npix, self.k, add_latent_dim,
                        add_latent_dim))
        nn.init.xavier_uniform_(self.weight_map_noise2)
        
        if self.noise_dim_high:
            self.weight_map_noisehigh = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh)
            
            self.weight_map_noisehigh2 = nn.Parameter(
                torch.empty(self.npix, self.k, self.noise_dim_high,
                            self.noise_dim_high))
            nn.init.xavier_uniform_(self.weight_map_noisehigh2)
            
        # ===== optional class scaling =================================
        if self.var_dims > 0:
            self.scale_raw = nn.Parameter(
                torch.zeros(self.num_max+1, self.var_dims))
        else:
            self.register_parameter("scale_raw", None)

        # ===== global-x encoder =======================================
        self.noise_x = noise_x
        self.mlp_x_reduce = make_mlp(
            (x_features-1) + noise_x, x_dim, hidden_x, depth_mlp_x)

        
        
        self.lags   = list(lags)
        self.n_prev = len(self.lags)

        self.add_latent_dim = (n_features if add_latent_dim is None
                               else add_latent_dim)
        self.noise_dim_prev = (n_features if noise_dim_prev is None
                               else noise_dim_prev)

        self.D_prev = (x_dim
                       + n_features                # current y_in average
                       + self.noise_dim_prev
                       + self.n_prev * self.keep_prev_feat)

        self.mlp_in_dim = x_dim + map_dim + noise_dim_high  + self.add_latent_dim
        self.final_mlp = make_mlp(self.mlp_in_dim,
                                   n_features,
                                   mlp_hidden,
                                   mlp_depth)
        self.latent_mlp = make_mlp(self.D_prev,
                                   self.add_latent_dim,
                                   mlp_hidden,
                                   mlp_depth)

    
    # -----------------------------------------------------------------
    # forward for a single step  (unchanged maths + extra latent)
    # -----------------------------------------------------------------
    def _forward_single(self, x, y_in, *, prev_list=None):
        """
        x      : (B, x_features)
        y_in   : (B, n_features, P)     – low-res input for this step
        prev_list : list[Tensor|None] of length n_prev,    each (B,F,P)
        """
        B, dev = x.size(0), x.device
        prev_list = prev_list or [None] * self.n_prev

        # ===== sparse-map from y_in ==================================
        noise_aug = torch.randn(B, self.noise_dim_aug,
                                self.npix, device=dev)
        if self.addconstant:
            with torch.no_grad():
                noise_aug[:,0,:] = 1.0
        y_cat   = torch.cat([y_in, noise_aug], 1)              # (B,C,P)
        ## weights (P,k,map,C)
        gather  = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :]   # (B,P,k,C)
        sparse  = torch.einsum("bpkn,pkmn->bpm",  gather, self.weight_map)        # (B,P,map)
        sparse = torch.einsum("bpkn,pkmn->bpm",   sparse[:,self.neighbor_indices,:], self.weight_map2)
        
        # ===== encode x (drop first col) =============================
        noise_vec = torch.randn(B, self.noise_x, device=dev)
        x_red = self.mlp_x_reduce(torch.cat([x[:, 1:], noise_vec], 1))   # (B,x_dim)
        x_exp = x_red.unsqueeze(1).expand(B, self.npix, self.x_dim)      # (B,P,x_dim)

        # ===== optional class-vector scaling =========================
        if self.var_dims > 0:
            cls   = x[:, 0].long().clamp_(0, self.num_max)
            scale = torch.exp(self.scale_raw[cls])                       # (B,var)
            #d     = min(self.var_dims, self.x_dim)
            d     = self.var_dims
            mask  = torch.ones_like(x_exp)
            mask[:, :, :d] = scale.view(B, 1, d)
            x_exp = x_exp * mask

        # ===== high-level noise vector ===============================
        noise_high = None
        if self.noise_dim_high:
            noise_high = (torch.randn(B, self.npix, self.noise_dim_high,
                                       device=dev))                # (B,P,noise)
            if self.addconstant:
                with torch.no_grad():
                    noise_high[..., 0] = 1.0
            noise_high = torch.einsum("bpkn,pkmn->bpm",
                                      noise_high[:, self.neighbor_indices, :] , self.weight_map_noisehigh)
            noise_high = torch.einsum("bpkn,pkmn->bpm",   noise_high[:,self.neighbor_indices,:], self.weight_map_noisehigh2)

        # ===== additional latent branch (lags) =======================
        
        
        if not any(p is None for p in prev_list) and torch.rand(1) < 0.995:

            noise_prev = torch.randn(B, self.npix,
                                        self.noise_dim_prev, device=dev)
                    
            y_naive = y_in.permute(0, 2, 1)

            prev_concat = torch.cat([ (p.permute(0, 2, 1) - y_naive)[:,:,range(self.keep_prev_feat)]
                                     for p in prev_list], 2)   # (B,P,F*n_prev)
            
            add_in = torch.cat([x_exp, y_naive, noise_prev, prev_concat], 2)  # (B,P,D_prev)
            
            add_latent = self.latent_mlp(add_in)                # (B,P,add_latent)
            #mask = torch.rand_like(tmp, dtype=torch.float32) < 0.95
            #add_latent = torch.where(mask, tmp, add_latent)
        else:
            add_latent = (torch.randn(B, self.npix,
                                self.add_latent_dim, device=dev))
            if self.addconstant:
                with torch.no_grad():
                    add_latent[:, :, 0] = 1.0
                        
            add_latent = torch.einsum("bpkn,pkmn->bpm",
                                    add_latent[:, self.neighbor_indices, :] , self.weight_map_noise)
            add_latent = torch.einsum("bpkn,pkmn->bpm",
                                    add_latent[:, self.neighbor_indices, :] , self.weight_map_noise2)
            # (B,P,add_latent)
        
        # ===== residual prediction ===================================
        parts = [x_exp, sparse, add_latent]
        if noise_high is not None:
            parts.append(noise_high)

        mlp_in  = torch.cat(parts, 2)                         # (B,P,Dtot)
        return  self.final_mlp(mlp_in).permute(0, 2, 1)   # (B,F,P)

    # -----------------------------------------------------------------
    # public forward (single step or autoregressive sequence)
    # -----------------------------------------------------------------
    def forward(
        self,
        x,              # (..., x_features)
        y_in,           # (..., n_features, P)
        *,
        dependence: bool = False,
        prev_tensors: dict[int, torch.Tensor] | None = None,
    ):
        # --- single step --------------------------------------------
        if not dependence:
            prev_list = None
            if prev_tensors is not None:
                prev_list = [prev_tensors.get(l) for l in self.lags]
            return self._forward_single(x, y_in, prev_list=prev_list)
       

        # --- autoregressive sequence --------------------------------
        orig_2d = x.dim() == 2
        if orig_2d:
            x, y_in = x.unsqueeze(1), y_in.unsqueeze(1)       # add time dim

        T = x.size(0)
        outs = []
        for t in range(T):
            prev_list = [outs[t - l] if t >= l else None for l in self.lags]
            outs.append(self._forward_single(x[t], y_in[t],
                                             prev_list=prev_list))

        out = torch.stack(outs, 0)                            # (T,B,F,P)
        return out.squeeze(1) if orig_2d else out
    
    
class LB2(nn.Module):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        nside_lo: int,
        nside_hi: int,
        *,
        n_features: int = 4,
        num_neighbors: int = 16,
        num_max: int = 35,       # highest cls index
    ):
        super().__init__()

        # ─── geometry ---------------------------------------------------
        self.nside_lo, self.nside_hi = nside_lo, nside_hi
        self.npix_hi = hp.nside2npix(nside_hi)
        self.npix_lo = hp.nside2npix(nside_lo) if nside_lo >= 1 else 0

        # ─── parameters -------------------------------------------------
        self.n_features = n_features
        self.num_cls    = num_max + 1        # include 0-class
        self.num_mon    = 13                 # 1 … 12

        # ─── neighbour indices (low→high) ------------------------------
        if self.npix_lo > 0:
            k = min(num_neighbors, self.npix_lo)
            lo_vec = np.vstack(hp.pix2vec(nside_lo, np.arange(self.npix_lo))).T
            hi_vec = np.vstack(hp.pix2vec(nside_hi, np.arange(self.npix_hi))).T
            idx_low = cKDTree(lo_vec).query(hi_vec, k=k)[1]             # (P_hi,k)
            self.register_buffer("neighbor_indices",
                                 torch.tensor(idx_low, dtype=torch.long))
            self.k = k
        else:                       # degenerate (no low-res grid)
            self.k = 0
            self.register_buffer("neighbor_indices",
                                 torch.empty(self.npix_hi, 0, dtype=torch.long))

        # ─── weight map: class × feature specific ----------------------
        if self.k > 0:
            self.weight_map = nn.Parameter(           # (C,F,P_hi,k)
                torch.zeros(self.num_cls, n_features, self.npix_hi, self.k)
            )
        else:
            self.weight_map = None

        # ─── biases: cls × mon × feat × pix ----------------------------
        if self.npix_lo > 0:
            self.bias_low = nn.Parameter(
                torch.zeros(self.num_cls, self.num_mon,
                            n_features, self.npix_lo)
            )                                        # (C,M,F,P_lo)
        else:
            self.bias_low = None

        self.bias_high = nn.Parameter(
            torch.zeros(self.num_cls, self.num_mon,
                        n_features, self.npix_hi)
        )                                            # (C,M,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def _upsample(self,
                  y_low: torch.Tensor,
                  cls:   torch.Tensor) -> torch.Tensor:
        """
        y_low : (B,F,P_lo)  
        cls   : (B,)        
        return: (B,F,P_hi)
        """
        B, P, _ = y_low.shape
        dev = y_low.device

        out = y_low.new_zeros(B, P, self.npix_hi)

        if self.k == 0:
            return out                     # nothing to up-sample

        idx = self.neighbor_indices        # (P_hi,k)

        for f in range(self.n_features):
            neigh = y_low[:, f][:, idx]                       # (B,P_hi,k)
            w     = self.weight_map[cls, f]                   # (B,P_hi,k)
            w = F.softmax(self.weight_map[cls, f], dim=-1) 
            out[:, f] = (neigh * w).sum(-1)                   # (B,P_hi)

        return out                                            # (B,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                   # (…,≥2) – only first two used
        y_low: torch.Tensor | None = None, # (…,F,P_lo) or None
        *,
        high: bool = True,                 # for bias-only mode
    ) -> torch.Tensor:

        # --- extract class & month ------------------------------------
        if x.dim() > 1:
            cls  = x[..., 0].long().clamp_(0, self.num_cls - 1)
            mon  = x[..., 1].long().clamp_(0, self.num_mon - 1)
        else:              
            raise RuntimeError("x needs at least first two dimensions (cls and month)")


        # flatten batch dims into leading dim “B”
        cls_flat = cls.view(-1)
        mon_flat = mon.view(-1)

        # ---------- bias–only path ------------------------------------
        if y_low is None:
            bias = (self.bias_high if high else self.bias_low)
            if bias is None:
                raise RuntimeError("Low-res grid absent → no low bias")
            out = bias[cls_flat, mon_flat]                 # (B,F,P)
            return out.view(*cls.shape, self.n_features,
                            -1)                            # restore batch-shape

        # ---------- full path (y_low given) ---------------------------
        if y_low.dim() != 3:
            raise ValueError("y_low must be 3-D (B,F,P_lo)")

        if self.bias_low is None:
            raise RuntimeError("Model was initialised without low-res grid")

        B = y_low.size(0)

        low_bias = self.bias_low[cls_flat, mon_flat].view_as(y_low)  # (B,F,P_lo)
        y_low_db = y_low - low_bias                                  # debiased

        up = self._upsample(y_low_db, cls_flat)                      # (B,F,P_hi)

        high_bias = self.bias_high[cls_flat, mon_flat]               # (B,F,P_hi)
        return up + high_bias  
 
class LB2mini(nn.Module):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        nside_lo: int,
        nside_hi: int,
        *,
        n_features: int = 4,
        num_neighbors: int = 9,
    ):
        super().__init__()

        # ─── geometry ---------------------------------------------------
        self.nside_lo, self.nside_hi = nside_lo, nside_hi
        self.npix_hi = hp.nside2npix(nside_hi)
        self.npix_lo = hp.nside2npix(nside_lo) if nside_lo >= 1 else 0

        # ─── parameters -------------------------------------------------
        self.n_features = n_features
        self.num_mon    = 13                 # 1 … 12

        # ─── neighbour indices (low→high) ------------------------------
        if self.npix_lo > 0:
            k = min(num_neighbors, self.npix_lo)
            lo_vec = np.vstack(hp.pix2vec(nside_lo, np.arange(self.npix_lo))).T
            hi_vec = np.vstack(hp.pix2vec(nside_hi, np.arange(self.npix_hi))).T
            idx_low = cKDTree(lo_vec).query(hi_vec, k=k)[1]             # (P_hi,k)
            self.register_buffer("neighbor_indices",
                                 torch.tensor(idx_low, dtype=torch.long))
            self.k = k
        else:                       # degenerate (no low-res grid)
            self.k = 0
            self.register_buffer("neighbor_indices",
                                 torch.empty(self.npix_hi, 0, dtype=torch.long))

        # ─── weight map: class × feature specific ----------------------
        if self.k > 0:
            self.weight_map = nn.Parameter(           # (F,P_hi,k)
                torch.zeros(n_features, self.npix_hi, self.k)
            )
        else:
            self.weight_map = None

        # ─── biases: cls × mon × feat × pix ----------------------------
        if self.npix_lo > 0:
            self.bias_low = nn.Parameter(
                torch.zeros(self.num_mon,
                            n_features, self.npix_lo)
            )                                        # (C,M,F,P_lo)
        else:
            self.bias_low = None

        self.bias_high = nn.Parameter(
            torch.zeros(self.num_mon,
                        n_features, self.npix_hi)
        )                                            # (C,M,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def _upsample(self,
                  y_low: torch.Tensor) -> torch.Tensor:
        
        B, P, _ = y_low.shape
        dev = y_low.device

        out = y_low.new_zeros(B, P, self.npix_hi)

        if self.k == 0:
            return out                     # nothing to up-sample

        idx = self.neighbor_indices        # (P_hi,k)

        for f in range(self.n_features):
            neigh = y_low[:, f][:, idx]                       # (B,P_hi,k)
            w     = self.weight_map[f]                   # (B,P_hi,k)
            w = F.softmax(self.weight_map[f], dim=-1) 
            out[:, f] = (neigh * w).sum(-1)                   # (B,P_hi)

        return out                                            # (B,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                   # (…,≥2) – only first two used
        y_low: torch.Tensor | None = None, # (…,F,P_lo) or None
        *,
        high: bool = True,                 # for bias-only mode
    ) -> torch.Tensor:

        # --- extract class & month ------------------------------------
        if x.dim() > 1:
            mon  = x[..., 1].long().clamp_(0, self.num_mon - 1)
        else:              
            raise RuntimeError("x needs at least first two dimensions (cls and month)")

        # flatten batch dims into leading dim “B”
        mon_flat = mon.view(-1)

        # ---------- bias–only path ------------------------------------
        if y_low is None:
            bias = (self.bias_high if high else self.bias_low)
            if bias is None:
                raise RuntimeError("Low-res grid absent → no low bias")
            out = bias[mon_flat]                 # (B,F,P)
            return out # out.view(*cls.shape, self.n_features,-1)                            # restore batch-shape

        # ---------- full path (y_low given) ---------------------------
        if y_low.dim() != 3:
            raise ValueError("y_low must be 3-D (B,F,P_lo)")

        if self.bias_low is None:
            raise RuntimeError("Model was initialised without low-res grid")

        B = y_low.size(0)
        low_bias = self.bias_low[mon_flat].view_as(y_low)  # (B,F,P_lo)
        y_low_db = y_low - low_bias                                  # debiased
        up = self._upsample(y_low_db)                      # (B,F,P_hi)
        high_bias = self.bias_high[mon_flat]               # (B,F,P_hi)
        return up + high_bias  

from typing import Optional, Dict, Any, Callable

