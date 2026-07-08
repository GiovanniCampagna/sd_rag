"""
HYBRID ontology for the ISP-18 (refuelling, Milan Bergamo) corpus, grounded in
the AISS ontology (Avio Information Semantic Schema, Al Masoud/Arazzi/Nocera et
al., University of Pavia — file AISS_Ontology-1.pdf).

Provenance / authoritative structure (from the AISS paper)
----------------------------------------------------------
- Event  (core class, extends ioe:Activity) specialises into ground operations
  (RefuellingEvent, CheckParkingEvent, AircraftParkingEvent, ...). Properties:
      aiss:takesPlaceAt  -> exactly one Site
      aiss:involves      -> physical objects + human agents + smart objects
      aiss:prerequires   -> other Event(s)  (time dependency: E1 needs E2..EN first)
- Site  (ioe:Site extends bot:Site): Airport bot:containsZone apron; apron
  containsZone stand + fuel-safety-zone; stand connected to gate.
- Organization (ORG): agent membership reifies agent+organization+role; roles
  subclass org:Role (FuelSafetySupervisor, ControlTowerOperator, RampAgent...).
- Flight (extends ioe:Process): origin/destination airport, aircraft, passengers.
- GSE: PushbackTractor, StairTruck, Bus, Ambulift, (refuelling truck/dispenser)...

The paper defines the T-Box (schema). The ISP-18 corpus and the professor's
reference diagram supply the A-Box surface forms (Italian) and a few concepts the
schema leaves implicit for this scenario (VVF fire service, COS, fuel spill
emergency, the BGY apron/stand instances). Those are tagged `isp18:Extension`.

This is the defining trait of the HYBRID method: WHICH entities and relations
exist is decided here, from the ontology — not by an LLM reading the text.
"""

from typing import List, Tuple, TypedDict


class OntologyConcept(TypedDict):
    id: str             # stable concept id (used as EntityNode ID)
    label: str          # human-readable name
    type: str           # AISS class (CURIE), or "isp18:Extension" if added on top
    aliases: List[str]  # surface forms found in the ISP-18 corpus (lowercased match)
    gloss: str          # short description, embedded for query->entity seeding


class Ontology(TypedDict):
    concepts: List[OntologyConcept]
    relations: List[Tuple[str, str, str]]  # (src_id, relation, dst_id); undirected for nav


# ---------------------------------------------------------------------------
# CONCEPTS — `type` is the AISS class the concept instantiates.
# ---------------------------------------------------------------------------

_CONCEPTS: List[OntologyConcept] = [

    # === EVENTS (aiss:Event subclasses) =====================================
    {
        'id': 'refuelling_event',
        'label': 'Rifornimento carburante',
        'type': 'aiss:RefuellingEvent',
        'aliases': ['rifornimento', 'rifornimenti', 'erogazione carburante', 'erogazione',
                    'refuelling', 'fuelling', 'defueling', 'travaso', 'rifornimento con passeggeri'],
        'gloss': "Operazione di rifornimento di carburante all'aeromobile, con o senza passeggeri a bordo.",
    },
    {
        'id': 'check_parking_event',
        'label': 'Controllo piazzola (pre-arrivo)',
        'type': 'aiss:CheckParkingEvent',
        'aliases': ['controllo piazzola', 'check parking', 'ispezione piazzola', 'fod', 'foreign object debris',
                    'verifica della piazzola', 'controllo pre-arrivo'],
        'gloss': "Verifica della piazzola prima dell'arrivo: rilevamento ostacoli (FOD) e stato di occupazione.",
    },
    {
        'id': 'aircraft_parking_event',
        'label': 'Parcheggio aeromobile',
        'type': 'aiss:AircraftParkingEvent',
        'aliases': ['parcheggio', 'sosta', 'parking', 'posizionamento aeromobile', 'docking', 'stop and go'],
        'gloss': "Posizionamento e sosta dell'aeromobile nella piazzola assegnata.",
    },
    {
        'id': 'passenger_transfer_event',
        'label': 'Imbarco/sbarco passeggeri',
        'type': 'aiss:PassengerTransferEvent',
        'aliases': ['imbarco', 'sbarco', 'boarding', 'trasferimento passeggeri', 'salita a bordo', 'discesa'],
        'gloss': "Imbarco e sbarco dei passeggeri durante il turnaround.",
    },
    {
        'id': 'emergenza',
        'label': 'Emergenza (sversamento/incendio)',
        'type': 'isp18:Extension',
        'aliases': ['emergenza', 'emergenze', 'incendio', 'sversamento', 'spandimento', 'sgombero',
                    'evacuazione', 'allarme', 'fuga di carburante', 'piano di emergenza'],
        'gloss': "Situazioni di emergenza nel rifornimento e piano di emergenza aeroportuale: sversamento di carburante, incendio, evacuazione, incidente.",
    },

    # === PHYSICAL OBJECTS (aiss:involves) ===================================
    {
        'id': 'aircraft',
        'label': 'Aeromobile',
        'type': 'aiss:Aircraft',
        'aliases': ['aeromobile', 'aeromobili', 'aereo', 'aerei', 'velivolo', 'velivoli', 'aircraft', 'a/m'],
        'gloss': "L'aeromobile fisico oggetto del rifornimento, parcheggiato nella piazzola.",
    },
    {
        'id': 'refuelling_truck',
        'label': 'Mezzo di rifornimento',
        'type': 'aiss:GSE',
        'aliases': ['autobotte', 'autobotti', 'automezzo di rifornimento', 'mezzo rifornitore', 'autocisterna',
                    'dispenser', 'mezzo di rifornimento', 'refuelling truck'],
        'gloss': "Automezzo (autobotte o dispenser), un GSE, che eroga il carburante in piazzola.",
    },
    {
        'id': 'gse',
        'label': 'Mezzi di assistenza a terra (GSE)',
        'type': 'aiss:GSE',
        'aliases': ['gse', 'ground support equipment', 'mezzi di assistenza', 'attrezzature di rampa',
                    'mezzi di rampa', 'pushback', 'ambulift', 'nastro bagagli'],
        'gloss': "Ground Support Equipment: mezzi e attrezzature di assistenza a terra in piazzola.",
    },
    {
        'id': 'avdgs',
        'label': 'AVDGS',
        'type': 'aiss:AVDGS',
        'aliases': ['avdgs', 'advanced visual docking guidance system', 'sistema di guida', 'docking guidance'],
        'gloss': "Advanced Visual Docking Guidance System: guida l'aeromobile al parcheggio in piazzola.",
    },
    {
        'id': 'carburante',
        'label': 'Carburante',
        'type': 'isp18:Extension',
        'aliases': ['carburante', 'combustibile', 'jet a1', 'jet a-1', 'jet a 1', 'kerosene', 'avio'],
        'gloss': "Il carburante avio (Jet A-1) erogato all'aeromobile durante il rifornimento.",
    },

    # === HUMAN AGENTS & ROLES (org:Role) ====================================
    {
        'id': 'ramp_agent',
        'label': 'Agente di rampa',
        'type': 'aiss:RampAgent',
        'aliases': ['agente di rampa', 'agenti di rampa', 'rampa', 'ramp agent', 'addetto di rampa'],
        'gloss': "Operatore di rampa che coordina le attività a terra in piazzola.",
    },
    {
        'id': 'fuel_safety_supervisor',
        'label': 'Responsabile del rifornimento',
        'type': 'aiss:FuelSafetySupervisor',
        'aliases': ['responsabile del rifornimento', 'responsabile rifornimento', 'supervisore',
                    'fuel safety supervisor', 'sorvegliante'],
        'gloss': "Persona individuata dall'Operatore Aereo che coordina e sorveglia le procedure di rifornimento.",
    },
    {
        'id': 'control_tower_operator',
        'label': 'Operatore torre / COS',
        'type': 'aiss:ControlTowerOperator',
        'aliases': ['torre di controllo', 'control tower', 'operatore di torre', 'cos',
                    'centro operativo di scalo', 'centro operativo', 'sala operativa',
                    'coordinamento di scalo', 'coordinamento scalo'],
        'gloss': "Coordinamento operativo dello scalo (torre di controllo / Centro Operativo di Scalo, COS, Coordinamento di Scalo).",
    },
    {
        'id': 'comandante',
        'label': 'Comandante / Equipaggio',
        'type': 'aiss:HumanAgent',
        'aliases': ['comandante', 'equipaggio', "membro dell'equipaggio", 'crew', 'cabin crew', 'flight crew'],
        'gloss': "Comandante o membro dell'equipaggio, possibili Responsabili del rifornimento.",
    },
    {
        'id': 'passeggeri',
        'label': 'Passeggeri',
        'type': 'aiss:Passenger',
        'aliases': ['passeggeri', 'passeggero', 'pax', 'a bordo'],
        'gloss': "I passeggeri a bordo o in imbarco/sbarco durante il rifornimento (rifornimento con passeggeri).",
    },

    # === ORGANIZATIONS (org:Organization) ===================================
    {
        'id': 'operatore_aereo',
        'label': 'Operatore Aereo',
        'type': 'org:Organization',
        'aliases': ['operatore aereo', 'operatori aerei', 'vettore', 'coa', 'certificato di operatore aereo',
                    'compagnia aerea'],
        'gloss': "Impresa di trasporto aereo commerciale in possesso di Certificato di Operatore Aereo (COA).",
    },
    {
        'id': 'handler',
        'label': 'Handler aeroportuale',
        'type': 'org:Organization',
        'aliases': ['handler', 'handler aeroportuale', 'prestatore di servizi', 'assistenza a terra',
                    'societa di handling', 'gestore aeroportuale'],
        'gloss': "Società prestatrice di servizi di assistenza a terra, incaricata dall'Operatore Aereo.",
    },
    {
        'id': 'vvf',
        'label': 'Vigili del Fuoco (VVF)',
        'type': 'isp18:Extension',
        'aliases': ['vvf', 'vigili del fuoco', 'vigili', 'servizio antincendio', 'antincendio',
                    'soccorso antincendio', 'sstas', 'distaccamento'],
        'gloss': "I Vigili del Fuoco aeroportuali, attivati/preavvisati durante il rifornimento (es. con passeggeri, o in emergenza).",
    },

    # === FLIGHT =============================================================
    {
        'id': 'flight',
        'label': 'Volo',
        'type': 'aiss:Flight',
        'aliases': ['volo', 'voli', 'flight'],
        'gloss': "Il volo da gestire durante il turnaround (aeromobile, passeggeri, origine/destinazione).",
    },

    # === SITES (ioe:Site / bot:Site) — generic + BGY A-Box instances ========
    {
        'id': 'stand',
        'label': 'Piazzola di sosta',
        'type': 'aiss:Stand',
        'aliases': ['piazzola', 'piazzole', 'stand', 'piazzola di sosta'],
        'gloss': "Piazzola di sosta aeromobili dove avviene il turnaround e il rifornimento.",
    },
    {
        'id': 'airport_bgy',
        'label': 'Aeroporto di Bergamo (BGY)',
        'type': 'aiss:Airport',
        'aliases': ['aeroporto', 'scalo', 'bergamo', 'orio al serio', 'bgy', 'milan bergamo'],
        'gloss': "Sedime aeroportuale di Milano Bergamo (Orio al Serio), in cui si applica la procedura ISP-18.",
    },
    {
        'id': 'apron_south',
        'label': 'Piazzale Sud (South Apron)',
        'type': 'aiss:Apron',
        'aliases': ['piazzale sud', 'south apron', 'apron sud'],
        'gloss': "Il piazzale sud dell'aeroporto, che contiene piazzole tra cui la 304.",
    },
    {
        'id': 'apron_north',
        'label': 'Piazzale Nord (North Apron)',
        'type': 'aiss:Apron',
        'aliases': ['piazzale nord', 'north apron', 'apron nord'],
        'gloss': "Il piazzale nord dell'aeroporto.",
    },
    {
        'id': 'stand_304',
        'label': 'Piazzola 304',
        'type': 'aiss:Stand',
        'aliases': ['piazzola 304', '304'],
        'gloss': "La piazzola 304 nel Piazzale Sud, con punti di sosta Stop 1 e Stop 2.",
    },
    {
        'id': 'stop_12',
        'label': 'Stop 1 / Stop 2',
        'type': 'aiss:Stand',
        'aliases': ['stop 1', 'stop 2', 'stop1', 'stop2', 'stop 1/2', 'stop1-2', 'stop 1-2'],
        'gloss': "I punti di sosta Stop 1 e Stop 2 della piazzola 304 (non abilitati al rifornimento con passeggeri).",
    },
]


# ---------------------------------------------------------------------------
# RELATION SKELETON  (src_id, relation, dst_id)
# AISS object properties (involves, takesPlaceAt, prerequires, containsZone) +
# ISP-18 procedural logic for the extension concepts (notifies, mayTrigger,
# notEnabledFor). Navigation treats edges as undirected.
# ---------------------------------------------------------------------------

_RELATIONS: List[Tuple[str, str, str]] = [

    # RefuellingEvent aiss:involves <physical objects / agents / orgs>
    ('refuelling_event', 'involves', 'aircraft'),
    ('refuelling_event', 'involves', 'refuelling_truck'),
    ('refuelling_event', 'involves', 'carburante'),
    ('refuelling_event', 'involves', 'ramp_agent'),
    ('refuelling_event', 'involves', 'fuel_safety_supervisor'),
    ('refuelling_event', 'involves', 'comandante'),
    ('refuelling_event', 'involves', 'operatore_aereo'),
    ('refuelling_event', 'involves', 'handler'),
    ('refuelling_event', 'involves', 'passeggeri'),

    # aiss:takesPlaceAt <Site>
    ('refuelling_event', 'takesPlaceAt', 'stand'),
    ('refuelling_event', 'takesPlaceAt', 'stand_304'),
    ('check_parking_event', 'takesPlaceAt', 'stand'),
    ('aircraft_parking_event', 'takesPlaceAt', 'stand'),
    ('passenger_transfer_event', 'takesPlaceAt', 'stand'),

    # aiss:prerequires <Event> (time dependency)
    ('refuelling_event', 'prerequires', 'aircraft_parking_event'),
    ('aircraft_parking_event', 'prerequires', 'check_parking_event'),
    ('passenger_transfer_event', 'prerequires', 'aircraft_parking_event'),

    # ISP-18 notification / emergency logic
    ('refuelling_event', 'notifies', 'vvf'),               # refuelling (with pax) preavvisa i VVF
    ('refuelling_event', 'notifies', 'control_tower_operator'),
    ('refuelling_event', 'mayTrigger', 'emergenza'),
    ('emergenza', 'notifies', 'vvf'),
    ('emergenza', 'notifies', 'control_tower_operator'),
    ('stop_12', 'notEnabledFor', 'refuelling_event'),      # "Stop 1/2 non abilitata" (diagramma)

    # Site topology (bot:containsZone / connection) — matches the diagram
    ('airport_bgy', 'containsZone', 'apron_south'),
    ('airport_bgy', 'containsZone', 'apron_north'),
    ('apron_south', 'containsZone', 'stand_304'),
    ('stand_304', 'hasStop', 'stop_12'),
    ('stand_304', 'isA', 'stand'),                         # instance -> generic stand concept

    # Spatial co-location at the stand (ioe:locatedIn)
    ('aircraft', 'locatedIn', 'stand'),
    ('refuelling_truck', 'locatedIn', 'stand'),
    ('avdgs', 'locatedIn', 'stand'),
    ('ramp_agent', 'locatedIn', 'stand'),
    ('avdgs', 'detects', 'aircraft_parking_event'),

    # Organisation / responsibility chain (ORG)
    ('operatore_aereo', 'designates', 'fuel_safety_supervisor'),
    ('operatore_aereo', 'employs', 'comandante'),
    ('comandante', 'canBe', 'fuel_safety_supervisor'),
    ('handler', 'operates', 'refuelling_truck'),
    ('handler', 'employs', 'ramp_agent'),

    # Flight links (aiss:hasAircraft / hasPassenger / origin-destination)
    ('flight', 'hasAircraft', 'aircraft'),
    ('flight', 'hasPassenger', 'passeggeri'),
    ('flight', 'atAirport', 'airport_bgy'),
    ('aircraft', 'carries', 'passeggeri'),
]


def get_ontology() -> Ontology:
    """Return the hybrid AISS-grounded ontology (concepts + relation skeleton)."""
    return {'concepts': _CONCEPTS, 'relations': _RELATIONS}
