"""Two-bank fixed-support bootstrap with a stationary real-data partition.

Only two consecutive, source-bound native D minibatches can confirm support
identities. The bootstrap requires equal MST group counts, reciprocal unique
nearest pairing, and a strict geometric separation margin. Afterward the
reference centers freeze: each real sample updates at most one identity in
its fixed nearest-reference cell. No new identity is born or deleted.

This is conditional fixed-target support learning, not proof that two finite
Gaussian banks reveal every component or that a target shift can be tracked.
"""

import hashlib
import math
import string
from copy import deepcopy

import torch

from reports.toy100.sample_group_anchor import mst_groups


SCHEMA = "two-bank-fixed-support-v1"
STATE_KEYS = {"schema", "last_bank_id", "last_bank_sha256", "pending", "confirmed",
              "confirmed_sums", "confirmed_counts", "confirmed_squared_norm_sums",
              "reference_centers", "fixed_half_separation", "accepted_samples_total",
              "rejected_samples_total", "confirmation"}
STAT_KEYS = {"bank_id", "bank_sha256", "sums", "counts", "squared_norm_sums"}


def _digest(value):
    return (isinstance(value,str) and len(value)==64
            and all(character in string.hexdigits for character in value))


def _clone_stats(value):
    return dict(bank_id=value["bank_id"], bank_sha256=value["bank_sha256"],
                sums=[x.clone() for x in value["sums"]], counts=list(value["counts"]),
                squared_norm_sums=[x.clone() for x in value["squared_norm_sums"]])


def _validate_stats(value):
    if set(value) != STAT_KEYS:
        raise ValueError("bank statistics lack required fields")
    if type(value["bank_id"]) is not int or value["bank_id"] < 1:
        raise ValueError("invalid bank id")
    if not _digest(value["bank_sha256"]):
        raise ValueError("invalid bank digest")
    sums, counts, sq = value["sums"], value["counts"], value["squared_norm_sums"]
    if not len(sums) == len(counts) == len(sq) or not sums:
        raise ValueError("bank statistics are empty or misaligned")
    for total,n,square in zip(sums,counts,sq):
        if (not isinstance(total,torch.Tensor) or total.shape!=(2,) or total.dtype!=torch.float64
                or not bool(torch.isfinite(total).all()) or type(n) is not int or n<1
                or not isinstance(square,torch.Tensor) or square.shape!=()
                or square.dtype!=torch.float64 or not bool(torch.isfinite(square))):
            raise ValueError("invalid group sufficient statistics")
        lower=float(total.square().sum())/n
        if float(square)+1e-8*max(1.,lower)<lower:
            raise ValueError("negative group variance")


def _min_separation(centers):
    if len(centers)<2:
        return None
    distances=torch.cdist(centers,centers)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min())


def _bank(real,bank_id):
    centers,grouping=mst_groups(real)
    members=grouping["member_indices"]
    value=dict(bank_id=bank_id,
        bank_sha256=hashlib.sha256(real.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        sums=[real[row].double().sum(0) for row in members],
        counts=[len(row) for row in members],
        squared_norm_sums=[real[row].double().square().sum() for row in members])
    _validate_stats(value)
    return centers,value,dict(groups=len(centers),min_separation=_min_separation(centers),
                              largest_mst_gap=grouping["largest_additive_gap"])


def _pair(first,second):
    a=torch.stack([total/n for total,n in zip(first["sums"],first["counts"])])
    b=torch.stack([total/n for total,n in zip(second["sums"],second["counts"])])
    if len(a)!=len(b) or len(a)<2:
        return None,dict(reason="unequal_or_single_group_counts",first_groups=len(a),second_groups=len(b))
    sa,sb=_min_separation(a),_min_separation(b)
    distances=torch.cdist(a,b)
    forward=distances.argmin(1).tolist()
    backward=distances.argmin(0).tolist()
    if len(set(forward))!=len(a) or any(backward[j]!=i for i,j in enumerate(forward)):
        return None,dict(reason="nonreciprocal_or_nonunique_pairing",first_groups=len(a),second_groups=len(b))
    max_pair=max(float(distances[i,j]) for i,j in enumerate(forward))
    margin=min(sa,sb)-2*max_pair
    if not math.isfinite(margin) or margin<=0:
        return None,dict(reason="nonpositive_separation_margin",first_groups=len(a),
                         second_groups=len(b),max_pair_distance=max_pair,min_separation=min(sa,sb),margin=margin)
    return forward,dict(reason="CONFIRMED",first_groups=len(a),second_groups=len(b),
                        max_pair_distance=max_pair,min_separation=min(sa,sb),margin=margin,
                        pairing=forward)


class TwoBankFixedSupportMemory:
    def __init__(self,*,expected_first_bank_id=1):
        if type(expected_first_bank_id) is not int or expected_first_bank_id<1:
            raise ValueError("invalid first bank id")
        self.expected_first_bank_id=expected_first_bank_id
        self.last_bank_id=None
        self.last_bank_sha256=None
        self.pending=None
        self.confirmed=False
        self.confirmed_sums=[]
        self.confirmed_counts=[]
        self.confirmed_squared_norm_sums=[]
        self.reference_centers=[]
        self.fixed_half_separation=None
        self.accepted_samples_total=0
        self.rejected_samples_total=0
        self.confirmation=None

    def state_dict(self):
        return dict(schema=SCHEMA,last_bank_id=self.last_bank_id,
            last_bank_sha256=self.last_bank_sha256,
            pending=None if self.pending is None else _clone_stats(self.pending),
            confirmed=self.confirmed,
            confirmed_sums=[x.clone() for x in self.confirmed_sums],
            confirmed_counts=list(self.confirmed_counts),
            confirmed_squared_norm_sums=[x.clone() for x in self.confirmed_squared_norm_sums],
            reference_centers=[x.clone() for x in self.reference_centers],
            fixed_half_separation=self.fixed_half_separation,
            accepted_samples_total=self.accepted_samples_total,
            rejected_samples_total=self.rejected_samples_total,
            confirmation=deepcopy(self.confirmation))

    def learner_state_dict(self):
        """Return the complete versioned state used by split-run checkpoints."""
        return self.state_dict()

    def load_state_dict(self,state):
        if set(state)!=STATE_KEYS or state["schema"]!=SCHEMA:
            raise ValueError("unsupported fixed-support learner state")
        bank_id=state["last_bank_id"]
        if type(bank_id) is not int or bank_id<1:
            raise ValueError("learner state lacks a completed bank")
        digest=state["last_bank_sha256"]
        if not _digest(digest):
            raise ValueError("invalid last-bank digest")
        if type(state["confirmed"]) is not bool:
            raise ValueError("invalid confirmation flag")
        pending=state["pending"]
        if pending is not None:
            _validate_stats(pending)
        if state["confirmed"]:
            if pending is not None or not isinstance(state["confirmation"],dict):
                raise ValueError("confirmed state has pending or missing proof")
            if (not len(state["reference_centers"])>=2
                    or not len(state["reference_centers"])==len(state["confirmed_sums"])==len(state["confirmed_counts"])==len(state["confirmed_squared_norm_sums"])):
                raise ValueError("confirmed groups are missing")
            verify=dict(bank_id=bank_id,bank_sha256=digest,sums=state["confirmed_sums"],
                        counts=state["confirmed_counts"],squared_norm_sums=state["confirmed_squared_norm_sums"])
            _validate_stats(verify)
            proof=state["confirmation"]
            proof_keys={"first_bank_id","second_bank_id","first_bank_sha256","second_bank_sha256",
                        "reason","first_groups","second_groups","max_pair_distance",
                        "min_separation","margin","pairing"}
            n_groups=len(state["reference_centers"])
            pairing=proof.get("pairing")
            geometry=(proof.get("max_pair_distance"),proof.get("min_separation"),proof.get("margin"))
            if (set(proof)!=proof_keys or proof["reason"]!="CONFIRMED"
                    or type(proof["first_bank_id"]) is not int
                    or type(proof["second_bank_id"]) is not int
                    or proof["first_bank_id"]+1!=proof["second_bank_id"]
                    or not 1<=proof["first_bank_id"]<proof["second_bank_id"]<=bank_id
                    or any(not _digest(proof[key])
                           for key in ("first_bank_sha256","second_bank_sha256"))
                    or proof["first_bank_sha256"]==proof["second_bank_sha256"]
                    or type(proof["first_groups"]) is not int or proof["first_groups"]!=n_groups
                    or type(proof["second_groups"]) is not int or proof["second_groups"]!=n_groups
                    or type(pairing) is not list or any(type(x) is not int for x in pairing)
                    or sorted(pairing)!=list(range(n_groups))
                    or any(type(x) is not float or not math.isfinite(x) for x in geometry)
                    or geometry[0]<0 or geometry[1]<=0 or geometry[2]<=0
                    or geometry[2]!=geometry[1]-2*geometry[0]
                    or (bank_id==proof["second_bank_id"]
                        and digest!=proof["second_bank_sha256"])):
                raise ValueError("invalid two-bank confirmation certificate")
            refs=state["reference_centers"]
            if any(not isinstance(x,torch.Tensor) or x.shape!=(2,) or x.dtype!=torch.float64
                   or not bool(torch.isfinite(x).all()) for x in refs):
                raise ValueError("invalid frozen references")
            half=_min_separation(torch.stack(refs))/2
            if (not isinstance(state["fixed_half_separation"],float)
                    or not math.isfinite(half) or half<=0
                    or state["fixed_half_separation"]!=half):
                raise ValueError("invalid fixed support separation")
        elif (pending is None or pending["bank_id"]!=bank_id or pending["bank_sha256"]!=digest
              or state["confirmation"] is not None or state["reference_centers"]
              or state["confirmed_sums"] or state["confirmed_counts"]
              or state["confirmed_squared_norm_sums"]
              or state["fixed_half_separation"] is not None
              or state["accepted_samples_total"]!=0 or state["rejected_samples_total"]!=0):
            raise ValueError("unconfirmed state contains altered support or lacks pending bank")
        for key in ("accepted_samples_total","rejected_samples_total"):
            if type(state[key]) is not int or state[key]<0:
                raise ValueError("invalid sample accounting")
        if state["confirmed"] and state["accepted_samples_total"]!=sum(state["confirmed_counts"]):
            raise ValueError("accepted sample accounting differs from accumulated counts")
        clone={key:value for key,value in state.items()}
        clone["pending"]=None if pending is None else _clone_stats(pending)
        for key in ("confirmed_sums","confirmed_squared_norm_sums","reference_centers"):
            clone[key]=[x.clone() for x in state[key]]
        clone["confirmed_counts"]=list(state["confirmed_counts"])
        clone["confirmation"]=deepcopy(state["confirmation"])
        for key,value in clone.items():
            if key!="schema":setattr(self,key,value)

    def load_learner_state_dict(self,state):
        """Restore complete learner state before observing the next real bank."""
        self.load_state_dict(state)

    def centers(self):
        if not self.confirmed:
            raise RuntimeError("two distinct compatible real banks have not confirmed support")
        return torch.stack([total/n for total,n in zip(self.confirmed_sums,self.confirmed_counts)])

    @torch.no_grad()
    def observe(self,real,*,bank_id):
        if (real.ndim!=2 or real.shape[1]!=2 or real.dtype!=torch.float32
                or not bool(torch.isfinite(real).all())):
            raise ValueError("finite native float32 real bank required")
        if type(bank_id) is not int or bank_id<1:
            raise ValueError("invalid absolute bank id")
        if self.last_bank_id is None:
            if bank_id!=self.expected_first_bank_id:
                raise RuntimeError("empty memory cannot silently restart at a later update")
        elif bank_id!=self.last_bank_id+1:
            raise RuntimeError("native data-bank observation must be consecutive")
        digest=hashlib.sha256(real.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if digest==self.last_bank_sha256:
            raise RuntimeError("identical consecutive bank bytes cannot corroborate bootstrap")

        if not self.confirmed:
            _,bank,summary=_bank(real,bank_id)
            if self.pending is None:
                self.pending=bank
                self.last_bank_id=bank_id
                self.last_bank_sha256=digest
                return dict(status="PENDING_FIRST_BANK",bank=summary,confirmed=False,
                            bank_id=bank_id,bank_sha256=digest)
            pairing,proof=_pair(self.pending,bank)
            if pairing is None:
                self.pending=bank
                self.last_bank_id=bank_id
                self.last_bank_sha256=digest
                return dict(status="UNRESOLVED_REPLACED",bank=summary,comparison=proof,
                            confirmed=False,bank_id=bank_id,bank_sha256=digest)
            first=self.pending
            self.confirmed_sums=[first["sums"][i]+bank["sums"][j] for i,j in enumerate(pairing)]
            self.confirmed_counts=[first["counts"][i]+bank["counts"][j] for i,j in enumerate(pairing)]
            self.confirmed_squared_norm_sums=[first["squared_norm_sums"][i]+bank["squared_norm_sums"][j]
                                               for i,j in enumerate(pairing)]
            self.reference_centers=[x.clone() for x in self.centers_from_stats()]
            self.fixed_half_separation=_min_separation(torch.stack(self.reference_centers))/2
            self.accepted_samples_total=sum(self.confirmed_counts)
            self.rejected_samples_total=0
            self.confirmation=dict(first_bank_id=first["bank_id"],second_bank_id=bank_id,
                first_bank_sha256=first["bank_sha256"],second_bank_sha256=digest,**proof)
            self.pending=None
            self.confirmed=True
            self.last_bank_id=bank_id
            self.last_bank_sha256=digest
            return dict(status="CONFIRMED",bank=summary,comparison=proof,
                        confirmed=True,confirmed_groups=len(self.reference_centers),
                        fixed_half_separation=self.fixed_half_separation,
                        bank_id=bank_id,bank_sha256=digest)

        refs=torch.stack(self.reference_centers)
        distances=torch.cdist(real.double(),refs)
        nearest=distances.argmin(1)
        closest=distances.gather(1,nearest[:,None]).squeeze(1)
        accepted=closest<self.fixed_half_separation
        assigned=[]
        for i in range(len(refs)):
            values=real[(nearest==i)&accepted].double()
            assigned.append(len(values))
            if len(values):
                self.confirmed_sums[i]+=values.sum(0)
                self.confirmed_counts[i]+=len(values)
                self.confirmed_squared_norm_sums[i]+=values.square().sum()
        n_accepted=int(accepted.sum())
        self.accepted_samples_total+=n_accepted
        self.rejected_samples_total+=len(real)-n_accepted
        self.last_bank_id=bank_id
        self.last_bank_sha256=digest
        return dict(status="UPDATED_FIXED_PARTITION",confirmed=True,bank_id=bank_id,
            bank_sha256=digest,accepted=n_accepted,rejected=len(real)-n_accepted,
            assigned_counts=assigned,confirmed_groups=len(refs),
            fixed_half_separation=self.fixed_half_separation)

    def centers_from_stats(self):
        return torch.stack([total/n for total,n in zip(self.confirmed_sums,self.confirmed_counts)])
