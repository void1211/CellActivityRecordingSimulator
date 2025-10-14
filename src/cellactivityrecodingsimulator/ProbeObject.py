from probeinterface import Probe
from probeinterface import generate_multi_columns_probe
from pathlib import Path
import json
import numpy as np
from typing import Union
from .Contact import Contact

class ProbeObject(Probe):
    def __init__(self, contacts: list[Contact]):
        super().__init__()
        self.contacts = contacts

    @classmethod
    def load(cls, object: Union[Path, Probe, dict, None]) -> "ProbeObject":
        if object is None:
            return cls.default_probe()
        elif isinstance(object, Path):
            return cls.from_json(object)
        elif isinstance(object, Probe):
            return cls.from_Probe(object)
        elif isinstance(object, dict):
            return cls.from_dict(object)

    @classmethod
    def default_probe(cls):
        """デフォルトのプローブを生成する"""
        probe = generate_multi_columns_probe(
            num_columns=1,
            num_contact_per_column=16,
            xpitch=0,
            ypitch=50,
            contact_shapes="circle",
            contact_shape_params={"radius": 10},
        )
        probe.set_device_channel_indices(list(range(probe.get_contact_count())))
        return cls.from_Probe(probe)

    @classmethod
    def from_json(cls, object: Path) -> "ProbeObject":
        with open(object, "r") as f:
            data = json.load(f)
            contacts = [
                Contact.from_dict({
                    "id": data["id"][i],
                    "x": data["x"][i],
                    "y": data["y"][i],
                    "z": data["z"][i]
                })
                for i in range(len(data["id"]))
            ]
            return cls(contacts)

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeObject":
        contacts = [
            Contact.from_dict({
                "id": data["id"][i],
                "x": data["x"][i],
                "y": data["y"][i],
                "z": data["z"][i]
            })
            for i in range(len(data["id"]))
        ]
        return cls(contacts)

    @classmethod
    def from_Probe(cls, probe: Probe) -> "ProbeObject":
        contacts = []
        probe_dict = probe.to_dict()
        contact_positions = np.array(probe_dict["contact_positions"])
        contact_ids = np.array(probe_dict["contact_ids"])
        is_3d = True
        if contact_positions.shape[1] != 3:
            is_3d = False
        for i in range(len(contact_positions)):
            if is_3d:
                contact_data = {
                    "id": contact_ids[i],
                    "x": contact_positions[i][0],
                    "y": contact_positions[i][1],
                    "z": contact_positions[i][2]
                }
            else:
                contact_data = {
                    "id": contact_ids[i],
                    "x": contact_positions[i][0],
                    "y": contact_positions[i][1],
                    "z": 0
                }
            contacts.append(Contact.from_dict(contact_data))
        probe_object = ProbeObject(contacts)
        return probe_object

    def get_contacts_num(self):
        return len(self.contacts)

    def get_contacts(self):
        return self.contacts
