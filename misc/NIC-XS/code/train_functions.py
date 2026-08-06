import torch
import copy
import healpy as hp
import numpy as np
import torch
from help_functions import healpix_avg_pooling
import os, csv
import torch.nn.functional as F
import torch.nn as nn
import csv, os, torch, torch.nn.functional as F
from itertools import cycle
from torch.utils.data import DataLoader, random_split
from typing import Iterable, Sequence, List

class BiasSum(nn.Module):
    def __init__(self, model_b: nn.Module, model_c: nn.Module, first=False):
        super().__init__()
        self.model_b = model_b
        self.model_c = model_c
        self.first = first
        #Optional: if you do **not** want to train them any further
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, *, y_low=None, x=None, **kw):
        if not self.first:
            out_b = self.model_b(y_low=y_low, x=x, **kw)
            out_c = self.model_c(y_in=out_b, x=x, **kw)
        else:
            out_b = self.model_b(x=x, **kw)
            out_c = self.model_c(x=x, **kw)
        return out_b + out_c
    
           
def energy_loss_two_sample(y_true, y_gen1, y_gen2, verbose=False):
    
    # Compute differences
    diff1 = y_true - y_gen1
    diff2 = y_true - y_gen2
    diff12 = y_gen1 - y_gen2
    diff1 = diff1.reshape(diff1.shape[0],-1)
    diff2 = diff2.reshape(diff2.shape[0],-1)
    diff12 = diff12.reshape(diff12.shape[0],-1)

    # Compute norms
    s1 = 0.5*(torch.mean(torch.norm(diff1, dim=1)) + torch.mean(torch.norm(diff2, dim=1)) ) 
    s2 = torch.mean(torch.norm(diff12, dim=1))

    # Combine terms
    loss = s1 - 0.5*s2

    if verbose:
        print(f'Loss: {loss.item():.4f}, s1: {s1.item():.4f}, s2: {s2.item():.4f}')

    return loss, s1, s2



def train_l2s(
    model,
    modelbias,
    train_loader,
    test_loader,
    downs,
    *,
    variables,          # indices kept for the loss / high-res channels
    nside_hi: int = 2,
    num_epochs: int = 10000,
    num_lag: int = 2000,
    lr: float = 1e-4,
    every_epoch: int = 5,
    device=torch.device("cuda"),
    save_dir: str = "checkpoints",
    only_test: bool = False,
    basic_mode: bool = True,
    checkpoint = None,
    supplyx=True,
    addnoise=0.001,
    lambdapen=10,
    det=False
):
    

    # ─── file & grid names ───────────────────────────────────────────────
    nside_lo = nside_hi // 2
    y_prev_name = f"prev{nside_hi}"
    y_prev_low_name = f"prev{nside_lo}"
    y_high_name = f"y{nside_hi}"
    y_low_name  = f"y{nside_lo}"
    os.makedirs(save_dir, exist_ok=True)

    if only_test:
        num_epochs = 1

    model = model.to(device).train()
    modelbias = modelbias.to(device).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay =0.001)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
        
    # ─── determine model options ────────────────────────────────────────
    
    needed_lags = model.lags

    # ─── input-combination definitions (same semantics as before) ───────
    combos = [{"name": "x", "inputs": {"x"}}] 

    def _get_y_low(batch, addnoise):
        y_low = batch[y_low_name].to(device, non_blocking=True)[:, variables, :]
        if addnoise>0:
            y_low = y_low + addnoise*torch.randn_like(y_low, device=device)
        return y_low

    # ─── bookkeeping for “best” model ───────────────────────────────────
    best_loss, best_epoch = float("inf"), start_epoch
    epoch = start_epoch 
    
    # ─────────────────────────── TRAIN / EVAL LOOP ──────────────────────
    while (epoch < num_epochs) and (best_epoch > epoch - num_lag):

        # ========== TRAIN ================================================
        if not only_test:
            model.train()
            losses_train = {c["name"]: 0.0 for c in combos}

            for batch in train_loader:
                optimizer.zero_grad()

                x       = batch["X"].to(device, non_blocking=True)
                y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                y_low   = _get_y_low(batch, addnoise)
                ####
                y_low_up = modelbias(y_low=y_low, x=x).detach()
                #####
                total_loss = 0.0
                for c in combos:
                    if supplyx:
                        model_in = {"x": x, "y_in": y_low_up}
                    else: 
                        model_in = {"y_in": y_low_up}

                    if not det:
                        y1, y2 = model(**model_in), model(**model_in)
                        loss, *_  = energy_loss_two_sample(y_high, y1, y2, False)
                        loss = loss + lambdapen * torch.sqrt(F.mse_loss(y_low, downs(y1)))
                    else:
                        y1 = model(**model_in)
                        loss  = F.mse_loss(y_high, y1)
                        loss = loss + lambdapen * torch.sqrt(F.mse_loss(y_low, downs(y1)))
                        
                    losses_train[c["name"]] += loss.item()
                    total_loss += loss

                if total_loss == 0.0:
                    continue
                total_loss.backward(); optimizer.step()

            # average over batches
            n_b = len(train_loader)
            for k in losses_train:
                losses_train[k] /= n_b
        else:
            losses_train = {c["name"]: 0.0 for c in combos}

        # ========== VALIDATION ===========================================
        if (epoch == 0) or ((epoch + 1) % every_epoch == 0):
            model.eval()
            losses_test = {c["name"]: 0.0 for c in combos}

            with torch.no_grad():
                for batch in test_loader:
                    x       = batch["X"].to(device, non_blocking=True)
                    y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                    y_low   = _get_y_low(batch, addnoise)
                    
                    ####
                    y_low_up = modelbias(y_low=y_low, x=x).detach()
                    #####
                    for c in combos:
                        if supplyx:
                            model_in = {"x": x, "y_in": y_low_up}
                        else: 
                            model_in = {"y_in": y_low_up}

                        if not det:
                            y1, y2 = model(**model_in), model(**model_in)
                            loss, *_  = energy_loss_two_sample(y_high, y1, y2, False)
                            loss = loss + lambdapen * torch.sqrt(F.mse_loss(y_low, downs(y1)))
                        else:
                            y1 = model(**model_in)
                            loss  = F.mse_loss(y_high, y1)
                            loss = loss + lambdapen * torch.sqrt(F.mse_loss(y_low, downs(y1)))
                     
                        losses_test[c["name"]] += loss.item()

            # average
            n_bt = len(test_loader)
            for k in losses_test:
                losses_test[k] /= n_bt

            # ---------- logging & checkpoints ----------------------------
            csv_path = os.path.join(save_dir, "losses.csv")
            first = not os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if first:
                    hdr = (["epoch"] +
                           [f"train_{k}" for k in losses_train] +
                           [f"test_{k}"  for k in losses_test])
                    w.writerow(hdr)
                w.writerow([epoch + 1] +
                           [losses_train[k] for k in losses_train] +
                           [losses_test[k]  for k in losses_test])

            print(f"Epoch {epoch + 1}")
            for k in losses_train:
                print(f"  train {k:>8}: {losses_train[k]:.4f}")
            for k in losses_test:
                print(f"  test  {k:>8}: {losses_test[k]:.4f}")

            #if (epoch + 1) % every_epoch == 0:
                # torch.save(model.state_dict(),
                #            os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt"))
                #torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

            total_test_loss = sum(losses_test.values())
            if total_test_loss < best_loss:
                best_loss, best_epoch = total_test_loss, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                # torch.save(
                #     {
                #         "model_state_dict": model.state_dict(),
                #         "optimizer_state_dict": optimizer.state_dict(),
                #         "epoch": epoch,
                #     },
                #     os.path.join(save_dir, "best_model_all.pt"),
                # )

            model.train()

        epoch += 1

    # ─── return best model ───────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model

def getmean(X):
    return torch.mean(X,dim=0, keepdim=True)

def train_l2shigh(
    model,
    modelbias,
    train_loader,
    test_loader,
    *,
    variables,          # indices kept for the loss / high-res channels
    nside_hi: int = 2,
    num_epochs: int = 10000,
    num_lag: int = 2000,
    lr: float = 1e-4,
    every_epoch: int = 5,
    device=torch.device("cuda"),
    save_dir: str = "checkpoints",
    only_test: bool = False,
    basic_mode: bool = True,
    checkpoint = None,
    supplyx=True,
    addnoise=0.001,
    lambdapen=10,
    det=False
):
    

    # ─── file & grid names ───────────────────────────────────────────────
    nside_lo = nside_hi // 2
    y_prev_name = f"prev{nside_hi}"
    y_prev_low_name = f"prev{nside_lo}"
    y_high_name = f"y{nside_hi}"
    y_low_name  = f"y{nside_lo}"
    os.makedirs(save_dir, exist_ok=True)

    if only_test:
        num_epochs = 1

    model = model.to(device).train()
    modelbias = modelbias.to(device).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay =0.001)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
        
    # ─── determine model options ────────────────────────────────────────
    
    needed_lags = model.lags

    # ─── input-combination definitions (same semantics as before) ───────
    combos = [{"name": "x", "inputs": {"x"}}] 

    def _get_y_low(batch, addnoise):
        y_low = batch[y_low_name].to(device, non_blocking=True)[:, variables, :]
        if addnoise>0:
            y_low = y_low + addnoise*torch.randn_like(y_low, device=device)
        return y_low

    # ─── bookkeeping for “best” model ───────────────────────────────────
    best_loss, best_epoch = float("inf"), start_epoch
    epoch = start_epoch 
    
    # ─────────────────────────── TRAIN / EVAL LOOP ──────────────────────
    while (epoch < num_epochs) and (best_epoch > epoch - num_lag):

        # ========== TRAIN ================================================
        if not only_test:
            model.train()
            losses_train = {c["name"]: 0.0 for c in combos}

            for batch in train_loader:
                optimizer.zero_grad()

                x       = batch["X"].to(device, non_blocking=True)
                y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                y_low   = _get_y_low(batch, addnoise)
                ####
                y_low_up = modelbias(y_low=y_low, x=x).detach()
                #####
                total_loss = 0.0
                for c in combos:
                    if supplyx:
                        model_in = {"x": x, "y_in": y_low_up}
                    else: 
                        model_in = {"y_in": y_low_up}

                    if not det:
                        y1, y2 = model(**model_in), model(**model_in)
                        loss, *_  = energy_loss_two_sample(y_high, y1, y2, False)
                    else:
                        y1 = model(**model_in)
                        loss  = F.mse_loss(y_high, y1)
                        
                    losses_train[c["name"]] += loss.item()
                    total_loss += loss

                if total_loss == 0.0:
                    continue
                total_loss.backward(); optimizer.step()

            # average over batches
            n_b = len(train_loader)
            for k in losses_train:
                losses_train[k] /= n_b
        else:
            losses_train = {c["name"]: 0.0 for c in combos}

        # ========== VALIDATION ===========================================
        if (epoch == 0) or ((epoch + 1) % every_epoch == 0):
            model.eval()
            losses_test = {c["name"]: 0.0 for c in combos}

            with torch.no_grad():
                for batch in test_loader:
                    x       = batch["X"].to(device, non_blocking=True)
                    y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                    y_low   = _get_y_low(batch, addnoise)
                    
                    ####
                    y_low_up = modelbias(y_low=y_low, x=x).detach()
                    #####
                    for c in combos:
                        if supplyx:
                            model_in = {"x": x, "y_in": y_low_up}
                        else: 
                            model_in = {"y_in": y_low_up}

                        if not det:
                            y1, y2 = model(**model_in), model(**model_in)
                            loss, *_  = energy_loss_two_sample(y_high, y1, y2, False)
                        else:
                            y1 = model(**model_in)
                            loss  = F.mse_loss(y_high, y1)
                     
                        losses_test[c["name"]] += loss.item()

            # average
            n_bt = len(test_loader)
            for k in losses_test:
                losses_test[k] /= n_bt

            # ---------- logging & checkpoints ----------------------------
            csv_path = os.path.join(save_dir, "losses.csv")
            first = not os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if first:
                    hdr = (["epoch"] +
                           [f"train_{k}" for k in losses_train] +
                           [f"test_{k}"  for k in losses_test])
                    w.writerow(hdr)
                w.writerow([epoch + 1] +
                           [losses_train[k] for k in losses_train] +
                           [losses_test[k]  for k in losses_test])

            print(f"Epoch {epoch + 1}")
            for k in losses_train:
                print(f"  train {k:>8}: {losses_train[k]:.4f}")
            for k in losses_test:
                print(f"  test  {k:>8}: {losses_test[k]:.4f}")

            total_test_loss = sum(losses_test.values())
            if total_test_loss < best_loss:
                best_loss, best_epoch = total_test_loss, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

            model.train()

        epoch += 1

    # ─── return best model ───────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model

def train_l3(
    model,
    modelbias,
    train_loader,
    test_loader,
    downs,
    *,
    variables,
    nside_hi: int = 2,
    num_epochs: int = 500,
    num_lag: int = 50,
    lr: float = 1e-4,
    every_epoch: int = 5,
    device=torch.device("cuda"),
    save_dir: str = "checkpoints",
    only_test: bool = False,
    basic_mode: bool = True,
    checkpoint=None,
    supplyx=True,
    addnoise=0.001,
    lambdapen=10.0,
    det=False,
    weight_decay: float = 0.0,
    penmean = 0.0, 
    nside_lo = None
):
    

    # ─── miscellaneous names & paths ──────────────────────────────────
    if nside_lo is None:
        nside_lo = nside_hi // 2
    y_prev_hi_name  = f"prev{nside_hi}"
    y_prev_lo_name  = f"prev{nside_lo}"
    y_high_name     = f"y{nside_hi}"
    y_low_name      = f"y{nside_lo}"
    os.makedirs(save_dir, exist_ok=True)

    # ─── number of epochs for pure-test mode ─────────────────────────
    if only_test:
        num_epochs = 1

    # ─── model / optimiser / checkpoint ──────────────────────────────
    model      = model.to(device).train()
    modelbias  = modelbias.to(device).eval()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    # ─── lag set (may be None) ───────────────────────────────────────
    needed_lags = model.lags

    # ─── input-combo list (unchanged) ────────────────────────────────
    combos = [{"name": "x", "inputs": {"x"}}] if basic_mode else [
        {"name": "x",        "inputs": {"x"}},
        {"name": "x_yprev",  "inputs": {"x", "y_prev"}},
    ]

    # ─── helper functions ────────────────────────────────────────────
    def _fetch_prev(batch, lag, hi=True):
        name = f"{y_prev_hi_name}_{lag}" if hi else f"{y_prev_lo_name}_{lag}"
        t = batch.get(name, None)
        return None if t is None else t.to(device, non_blocking=True)[:, variables, :]
    
    def _build_prev_dict(batch,hi=True):
        d, missing = {}, False
        if needed_lags is not None:
            for lag in needed_lags:
                t = _fetch_prev(batch, lag, hi=hi)
                d[lag] = t
                if t is None:
                    missing = True
        else:
            missing = True            
        return d, missing
    
    # ─── bookkeeping for the “best” checkpoint ───────────────────────
    best_loss, best_epoch = float("inf"), start_epoch
    epoch = start_epoch

    # ─────────────────────────── TRAIN / EVAL LOOP ───────────────────
    while (epoch < num_epochs) and (best_epoch > epoch - num_lag):
        
        # ========================= TRAIN =============================
        if not only_test:
            model.train()
            losses_tr = {c["name"]: 0.0 for c in combos}
            for batch in train_loader:
                optimizer.zero_grad()

                x       = batch["X"].to(device, non_blocking=True)
                y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                if nside_lo>0:
                    y_low   = batch[y_low_name].to(device, non_blocking=True)[:, variables, :]
                else:
                    y_low = None

                # lag handling ---------------------------------------
                prev_dict, missing_prev = _build_prev_dict(batch)
                
                # bias & debias current target -----------------------
                y_low_up = modelbias(x=x, y_low=y_low).detach()

                total = 0.0
                for c in combos:
                    # assemble model input --------------------------
                    model_in = {"y_in": y_low_up}
                    if supplyx:
                        model_in["x"] = x
                    if "y_prev" in c["inputs"]:
                        model_in["prev_tensors"] = prev_dict
                    

                    # forward + loss -------------------------------
                    if det:
                        y_pred = model(**model_in)
                        loss   = F.mse_loss(y_high, y_pred) \
                                 + lambdapen * torch.sqrt(
                                       F.mse_loss(y_low,
                                                  downs(y_pred)))
                    else:
                        y1, y2 = model(**model_in), model(**model_in)
                        
                        loss, *_ = energy_loss_two_sample(y_high, y1, y2, False)
                        if nside_lo>0:
                            loss += lambdapen * torch.sqrt( F.mse_loss(y_low,   downs(y1)))
                        if penmean > 0.0:
                            lossmu, *_   = energy_loss_two_sample(y_high.mean(dim=2), y1.mean(dim=2), y2.mean(dim=2), False)
                            loss = loss + penmean*lossmu
                    losses_tr[c["name"]] += loss.item()
                    total += loss

                if total == 0.0:
                    continue
                total.backward()
                optimizer.step()

            # mean over batches
            nb = len(train_loader)
            for k in losses_tr:
                losses_tr[k] /= nb
        else:
            losses_tr = {c["name"]: 0.0 for c in combos}

        # ======================= VALIDATION ==========================
        if (epoch == 0) or ((epoch + 1) % every_epoch == 0):
            model.eval()
            losses_te = {c["name"]: 0.0 for c in combos}

            with torch.no_grad():
                for batch in test_loader:
                    x       = batch["X"].to(device, non_blocking=True)
                    y_high  = batch[y_high_name].to(device, non_blocking=True)[:, variables, :]
                    if nside_lo>0:
                        y_low   = batch[y_low_name].to(device, non_blocking=True)[:, variables, :]
                    else:
                        y_low = None

                    prev_dict, missing_prev = _build_prev_dict(batch)

                    y_low_up = modelbias(x=x, y_low=y_low).detach()

                    for c in combos:
                        if "y_prev" in c["inputs"] and missing_prev:
                            continue

                        model_in = {"y_in": y_low_up}
                        if supplyx:
                            model_in["x"] = x
                        if "y_prev" in c["inputs"]:
                            model_in["prev_tensors"] = prev_dict
                       
                        if det:
                            y_pred = model(**model_in)
                            loss   = F.mse_loss(y_high, y_pred) \
                                     + lambdapen * torch.sqrt(
                                           F.mse_loss(y_low,
                                                      downs(y_pred)))
                        else:
                            y1, y2 = model(**model_in), model(**model_in)
                            loss, *_ = energy_loss_two_sample(y_high, y1, y2, False)
                            if nside_lo>0:
                                loss += lambdapen * torch.sqrt( F.mse_loss(y_low, downs(y1)))
                            if penmean > 0.0:
                                lossmu, *_   = energy_loss_two_sample(y_high.mean(dim=2), y1.mean(dim=2), y2.mean(dim=2), False)
                                loss = loss + penmean*lossmu

                        losses_te[c["name"]] += loss.item()

            # mean over test batches
            nb_te = len(test_loader)
            for k in losses_te:
                losses_te[k] /= nb_te

            # ---------- CSV / stdout --------------------------------
            csv_p = os.path.join(save_dir, "losses.csv")
            hdr_needed = not os.path.isfile(csv_p)
            with open(csv_p, "a", newline="") as f:
                w = csv.writer(f)
                if hdr_needed:
                    w.writerow(["epoch"] +
                               [f"train_{k}" for k in losses_tr] +
                               [f"test_{k}"  for k in losses_te])
                w.writerow([epoch + 1] +
                           [losses_tr[k] for k in losses_tr] +
                           [losses_te[k] for k in losses_te])

            print(f"Epoch {epoch + 1}")
            for k in losses_tr:
                print(f"  train {k:>8}: {losses_tr[k]:.4e}")
            for k in losses_te:
                print(f"  test  {k:>8}: {losses_te[k]:.4e}")

            # checkpoints -------------------------------------------
            # if (epoch + 1) % every_epoch == 0:
            #     torch.save(model.state_dict(),
            #                os.path.join(save_dir,
            #                             f"model_epoch_{epoch + 1}.pt"))

            total_val = sum(losses_te.values())
            if total_val < best_loss:
                best_loss, best_epoch = total_val, epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "best_model.pt"))
                # torch.save({"model_state_dict": model.state_dict(),
                #             "optimizer_state_dict": optimizer.state_dict(),
                #             "epoch": epoch},
                #            os.path.join(save_dir, "best_model_all.pt"))
            model.train()

        epoch += 1

    # ─── return best model ───────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model


import os, csv, torch

def train_lb2(
    model,
    train_loader,
    test_loader,
    *,
    variables,
    num_epochs: int  = 100000,
    num_lag:    int  = 50000, #50
    num_min = 50,
    lr:         float = 1e-3,
    every_epoch: int = 100,
    device = torch.device("cuda"),
    save_dir: str = "checkpoints",
    only_test: bool = False,
    checkpoint = None,
    addnoise: float = 1e-6,
):
    
    # ─── grid names ------------------------------------------------------
    ns_hi, ns_lo = model.nside_hi, model.nside_lo      # ns_lo==0 if none
    y_hi_nm = f"y{ns_hi}"
    y_lo_nm = f"y{ns_lo}" if ns_lo else None

    os.makedirs(save_dir, exist_ok=True)
    if only_test:
        num_epochs = 1

    # ─── COMBO definition -----------------------------------------------
    have_low = (ns_lo != 0)

    combos = []

    # --- up-sampling (only if low-res grid present) ----------------------
    if have_low:
        combos.append(dict(
            name="x",
            call=lambda x, yl: model(x=x, y_low=yl),   # full pipeline
            trg="high",
        ))

    # --- high-resolution bias (always) ----------------------------------
    combos.append(dict(
        name="xhigh",
        call=lambda x, _: model(x=x, y_low=None, high=True),
        trg="high",
    ))

    # --- low-resolution bias (only if low-res grid present) -------------
    if have_low:
        combos.append(dict(
            name="xlow",
            call=lambda x, _: model(x=x, y_low=None, high=False),
            trg="low",
        ))

    # ─── helper: fetch / perturb low-res field ---------------------------
    def _get_y_low(batch):
        if not have_low:
            return None
        yl = batch[y_lo_nm].to(device, non_blocking=True)[:, variables, :]
        if addnoise:
            yl = yl + addnoise * torch.randn_like(yl)
        return yl

    # ─── model / optimiser ----------------------------------------------
    model = model.to(device).train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    #opt   = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.99)
    
    # reload checkpoint ---------------------------------------------------
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    best_loss, best_epoch = float("inf"), start_epoch
    epoch = start_epoch

    # ─────────────────────────── TRAIN / EVAL LOOP ───────────────────────
    while epoch < num_epochs and (epoch < num_min or best_epoch > epoch - num_lag):

        # =========================== TRAIN ===============================
        if not only_test:
            model.train()
            loss_tr = {c["name"]: 0.0 for c in combos}

            for batch in train_loader:
                x   = batch["X"].to(device, non_blocking=True)
                yhi = batch[y_hi_nm].to(device, non_blocking=True)[:, variables, :]
                ylo = _get_y_low(batch)
                
               
                opt.zero_grad()
                total = 0.0

                for c in combos:
                    target = yhi if c["trg"] == "high" else ylo
                    if target is None:        # happens for "xlow" when no low-res
                        continue
                    pred  = c["call"](x, ylo)
                        
                    # csv_p = os.path.join(save_dir, "means.csv")
                    # with open(csv_p, "a", newline="") as f:
                    #     w = csv.writer(f)
                    #     w.writerow([target[(x[:,0]==7).nonzero(),8,:].mean()] +
                    #                [pred[(x[:,0]==7).nonzero(),8,:].mean().detach()] +
                    #             [(model.bias_high[7,:,8,:]).mean().detach()])
                        
                    loss  = F.mse_loss(pred, target)
                    loss.backward()
                    total += loss
                    loss_tr[c["name"]] += loss.item()

                if total != 0:
                    opt.step()

            # average train losses
            nb = len(train_loader)
            for k in loss_tr:
                loss_tr[k] /= nb
        else:
            loss_tr = {c["name"]: 0.0 for c in combos}

        # ========================== VALIDATION ===========================
        if (epoch == 0) or ((epoch + 1) % every_epoch == 0):
            model.eval()
            loss_te = {c["name"]: 0.0 for c in combos}

            with torch.no_grad():
                for batch in test_loader:
                    x   = batch["X"].to(device, non_blocking=True)
                    yhi = batch[y_hi_nm].to(device, non_blocking=True)[:, variables, :]
                    ylo = _get_y_low(batch)

                    for c in combos:
                        target = yhi if c["trg"] == "high" else ylo
                        if target is None:
                            continue
                        pred = c["call"](x, ylo)
                        loss_te[c["name"]] += F.mse_loss(pred, target).item()

            nb_te = len(test_loader)
            for k in loss_te:
                loss_te[k] /= nb_te

            # ---------- CSV / stdout ------------------------------------
            csv_p = os.path.join(save_dir, "losses.csv")
            hdr_needed = not os.path.isfile(csv_p)
            with open(csv_p, "a", newline="") as f:
                w = csv.writer(f)
                if hdr_needed:
                    w.writerow(["epoch"] +
                               [f"train_{k}" for k in loss_tr] +
                               [f"test_{k}"  for k in loss_te])
                w.writerow([epoch + 1] +
                           [loss_tr[k] for k in loss_tr] +
                           [loss_te[k] for k in loss_te])

            print(f"Epoch {epoch+1}")
            for k in loss_tr:
                print(f"  train {k:>5}: {loss_tr[k]:.4e}")
            for k in loss_te:
                print(f"  test  {k:>5}: {loss_te[k]:.4e}")

            # ---------- checkpoints -----------------------------------
            #if (epoch + 1) % every_epoch == 0:
                #torch.save(model.state_dict(),
                #           os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
                #torch.save(model.state_dict(),
                #           os.path.join(save_dir, "best_model.pt"))
                
            total_val = sum(loss_te.values())
            if total_val < best_loss:
                best_loss, best_epoch = total_val, epoch
            
            torch.save(model.state_dict(),  os.path.join(save_dir, "best_model.pt"))
                
                #torch.save({"model_state_dict": model.state_dict(),
                #            "optimizer_state_dict": opt.state_dict(),
                #            "epoch": epoch},
                #           os.path.join(save_dir, "best_model_all.pt"))
            model.train()

        epoch += 1

    # ─── return best model ----------------------------------------------
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model


def train_lb2f(
    model,
    train_loader,
    test_loader,
    *,
    variables,
    num_epochs: int  = 100000,
    num_lag:    int  = 50000, #50
    num_min = 50,
    lr:         float = 1e-4,
    every_epoch: int = 100,
    device = torch.device("cuda"),
    save_dir: str = "checkpoints",
    only_test: bool = False,
    checkpoint = None
):
    
    # ─── grid names ------------------------------------------------------
    ns_hi = model.nside_hi
    y_hi_nm = f"y{ns_hi}"

    os.makedirs(save_dir, exist_ok=True)
    if only_test:
        num_epochs = 1
    combos = []

    # --- high-resolution bias (always) ----------------------------------
    combos.append(dict(
        name="xhigh",
        call=lambda x, _: model(x=x),
        trg="high",
    ))

    # ─── model / optimiser ----------------------------------------------
    model = model.to(device).train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    # reload checkpoint ---------------------------------------------------
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    best_loss, best_epoch = float("inf"), start_epoch
    epoch = start_epoch

    # ─────────────────────────── TRAIN / EVAL LOOP ───────────────────────
    while epoch < num_epochs and (epoch < num_min or best_epoch > epoch - num_lag):

        # =========================== TRAIN ===============================
        if not only_test:
            model.train()
            loss_tr = {c["name"]: 0.0 for c in combos}

            for batch in train_loader:
                x   = batch["X"].to(device, non_blocking=True)
                yhi = batch[y_hi_nm].to(device, non_blocking=True)[:, variables, :]

                opt.zero_grad()
                total = 0.0

                for c in combos:
                    target = yhi 
                    pred  = c["call"](x)
                    loss  = F.mse_loss(pred, target)
                    loss.backward()
                    total += loss
                    loss_tr[c["name"]] += loss.item()

                if total != 0:
                    opt.step()

            # average train losses
            nb = len(train_loader)
            for k in loss_tr:
                loss_tr[k] /= nb
        else:
            loss_tr = {c["name"]: 0.0 for c in combos}

        # ========================== VALIDATION ===========================
        if (epoch == 0) or ((epoch + 1) % every_epoch == 0):
            model.eval()
            loss_te = {c["name"]: 0.0 for c in combos}

            with torch.no_grad():
                for batch in test_loader:
                    x   = batch["X"].to(device, non_blocking=True)
                    yhi = batch[y_hi_nm].to(device, non_blocking=True)[:, variables, :]

                    for c in combos:
                        target = yhi
                       
                        pred = c["call"](x)
                        loss_te[c["name"]] += F.mse_loss(pred, target).item()

            nb_te = len(test_loader)
            for k in loss_te:
                loss_te[k] /= nb_te

            # ---------- CSV / stdout ------------------------------------
            csv_p = os.path.join(save_dir, "losses.csv")
            hdr_needed = not os.path.isfile(csv_p)
            with open(csv_p, "a", newline="") as f:
                w = csv.writer(f)
                if hdr_needed:
                    w.writerow(["epoch"] +
                               [f"train_{k}" for k in loss_tr] +
                               [f"test_{k}"  for k in loss_te])
                w.writerow([epoch + 1] +
                           [loss_tr[k] for k in loss_tr] +
                           [loss_te[k] for k in loss_te])

            print(f"Epoch {epoch+1}")
            for k in loss_tr:
                print(f"  train {k:>5}: {loss_tr[k]:.4e}")
            for k in loss_te:
                print(f"  test  {k:>5}: {loss_te[k]:.4e}")

            total_val = sum(loss_te.values())
            if total_val < best_loss:
                best_loss, best_epoch = total_val, epoch
            
            torch.save(model.state_dict(),  os.path.join(save_dir, "best_model.pt"))
            model.train()

        epoch += 1

    # ─── return best model ----------------------------------------------
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model
