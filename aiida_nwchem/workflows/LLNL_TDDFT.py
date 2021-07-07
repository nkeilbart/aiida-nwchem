from aiida import load_profile, orm
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida.engine import submit, WorkChain, ToContext, if_, while_, append_
from aiida.common import AttributeDict

import numpy as np
import copy

NwchemCalculation = CalculationFactory('nwchem.nwchem')


class LLNLSpectroscopyWorkChain(WorkChain):
    """
    This work chain will contain six separate calculations.
    1) Charged atom with a cage of point charges (RHF)
    2) Ligand calculation with point charge in place of atom of interest (RHF)
    3) Full system pulling the previously computed movec files (RHF)
    4) SCF calculation with Unrestricted Hartree Fock calculation (UHF)
    5) DFT calculation including COSMO
    6) TDDFT calculation
    """ 

    @classmethod
    def define(cls,spec):
        # yapf: disable
        """
        Define the parameters and workflow
        """
        super().define(spec)
        spec.expose_inputs(NwchemCalculation, namespace = 'cage',
            namespace_options={'help': 'Inputs from the NwchemCalculation for cage calculation.'})
        spec.expose_inputs(NwchemCalculation, namespace = 'ligand',
            namespace_options={'help': 'Inputs from the NwchemCalculation for ligand only calculation.'})
        spec.expose_inputs(NwchemCalculation, namespace = 'full',
            namespace_options={'help': 'Inputs from the NwchemCalculation for combined cage and ligand calculation.'})
        spec.expose_inputs(NwchemCalculation, namespace = 'uhf',
            namespace_options={'help': 'Inputs from the NwchemCalculation for full with UHF settings.'})
        spec.expose_inputs(NwchemCalculation, namespace = 'dft',
            namespace_options={'help': 'Inputs from the NwchemCalculation for dft calculation.'})
        spec.expose_inputs(NwchemCalculation, namespace = 'tddft',
            namespace_options={'help': 'Inputs from the NwchemCalculation for tddft calculation.'})
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all calculations are deleted at end of workflow.')

        spec.outline(
            cls.run_cage,
            cls.check_cage,

            cls.run_ligand,
            cls.check_ligand,

            cls.run_full,
            cls.check_full,

            cls.run_uhf,
            cls.check_uhf,

            cls.run_dft,
            cls.check_dft,

            cls.run_tddft,
            cls.check_tddft,

            cls.results,
        )
        spec.expose_outputs(NwchemCalculation)
        spec.exit_code(401, 'NO_CHARGE_SPECIFIED',
            message='must specify a charge for the builder')
        spec.exit_code(410, 'NO_SPIN_MULT',
            message='must specify a spin multiplicity for the builder')
        spec.exit_code(402, 'NO_ATOM_FOUND',
            message='no atoms found when parsing structure for elements other than "H" and "O"')
        spec.exit_code(403, 'TOO_MANY_ATOMS_FOUND',
            message='found more than one atom in structure when parsing for cage calculation')
        spec.exit_code(404, 'CAGE_FAILED',
            message='cage calculation failed')
        spec.exit_code(405, 'LIGAND_FAILED',
            message='ligand calculation failed')
        spec.exit_code(406, 'FULL_FAILED',
            message='full calculation failed')
        spec.exit_code(407, 'UHF_FAILED',
            message='uhf calculation failed')
        spec.exit_code(408, 'DFT_FAILED',
            message='dft calculation failed')
        spec.exit_code(409, 'TDDFT_FAILED',
            message='tddft calculation failed')

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, charge=None, spin_mult=None, 
        cage_style='octahedra', overrides=None, **kwargs
    ):

        if charge == None:
            return cls.exit_codes.NO_CHARGE_SPECIFIED
        if spin_mult == None:
            return cls.exit_codes.NO_SPIN_MULT

        # spin_mult = 2 * (# of unpaired electrons/2) + 1
        spin_states = { 1 : 'singlet',
                        2 : 'doublet',
                        3 : 'triplet' }

        spin_state = spin_states[spin_mult]

        args = (code, structure)

        builder = cls.get_builder()

        # Find which atom
        StructureData = DataFactory('structure')
        cage_structure = StructureData()
        ligand_structure = StructureData()
        for site in structure.sites:
            site_dict = site.get_raw()
            kind = site_dict['kind_name']
            pos = site_dict['position']
            if kind != 'H' and kind != 'O':
                cage_kind = kind
                cage_pos = pos
                cage_structure.append_atom(symbols=kind,position=pos)
                ligand_charge = [[pos[0],pos[1],pos[2],charge]]
            elif kind == 'H' or kind == 'O':
                ligand_structure.append_atom(symbols=kind,position=pos)

        # Check that there is only one atom
        if len(cage_structure.sites) < 1:
            return cls.exit_codes.NO_ATOM_FOUND
        elif len(cage_structure.sites) > 1:
            return cls.exit_codes.TOO_MANY_ATOMS_FOUND
        else:
            builder.cage.structure = cage_structure
            builder.ligand.structure = ligand_structure
            
        # Setup cage charge based on cage_style
        cage_charge = cls.get_cage_charge(charge,cage_style)
        point_charges = cls.get_point_charges(cage_style,cage_pos)

        # Setup default metadata
        Dict = DataFactory('dict')
        metadata = {
            'options' : {
                'resources' : {
                    'num_machines' : 1
                },
                'max_wallclock_seconds' : 30*60,
                'queue_name' : 'pbatch',
                'account' : 'corrctl'
            }
        }
 

        # Setup caged parameters
        builder.cage.code = code
        builder.cage.metadata = copy.deepcopy(metadata)
        builder.cage.metadata['call_link_label'] = 'cage'
        builder.cage.parameters = Dict(dict={

            'basis spherical':{
                'H' : 'library aug-cc-pVDZ',
                'O' : 'library aug-cc-pVDZ',
                cage_kind : cls.cu_basis()
            },

            'set':{
                'tolguess': 1e-9
             },

            'charge' : cage_charge,

            'point_charges' : point_charges,

            'scf' : {
                spin_state:'',
                'maxiter' : 100,
                'vectors' : 'atomic output {0}.movecs'.format(cage_kind.lower())
            },

            'task' : 'scf energy'
        })

        builder.ligand.code = code
        builder.ligand.metadata = copy.deepcopy(metadata)
        builder.ligand.metadata['call_link_label'] = 'ligand'
        builder.ligand.parameters = Dict(dict={
            'basis spherical':{
                'H' : 'library aug-cc-pVDZ',
                'O' : 'library aug-cc-pVDZ',
                cage_kind : cls.cu_basis()
            },
            'set':{
                'tolguess': 1e-9
             },
            'charge' : charge,
            'point_charges' : ligand_charge,
            'scf' : {
                'singlet':'',
                'maxiter' : 100,
                'vectors' : 'atomic output ligand.movecs'
            },
            'task' : 'scf energy'
        })
        
        # Setup full calculation information
        builder.full.code = code
        builder.full.metadata = copy.deepcopy(metadata)
        builder.full.metadata['call_link_label'] = 'full'
        builder.full.structure = structure
        builder.full.parameters = Dict(dict={
            'basis spherical':{
                'H' : 'library aug-cc-pVDZ',
                'O' : 'library aug-cc-pVDZ',
                cage_kind : cls.cu_basis()
            },
            'set':{
                'tolguess': 1e-9
             },
            'charge' : charge,
            'scf' : {
                spin_state:'',
                'maxiter' : 100,
                'vectors' : 'input fragment {0}.movecs ligand.movecs output hf.movecs'.format(cage_kind.lower())
            },
            'task' : 'scf energy'
        })
           
        # Setup full calculation with UHF
        builder.uhf.code = code
        builder.uhf.metadata = copy.deepcopy(metadata)
        builder.uhf.metadata['call_link_label'] = 'uhf'
        builder.uhf.structure = structure
        builder.uhf.parameters = Dict(dict={
            'restart' : True,
            'scf' : {
                'vectors' : 'input hf.movecs output uhf.movecs',
                '{0}; uhf'.format(spin_state) : '',
                'maxiter' : 100
            },
            'task' : 'scf energy'
        })    
        
        # DFT parameters
        builder.dft.code = code
        builder.dft.metadata = copy.deepcopy(metadata)
        builder.dft.metadata['call_link_label'] = 'dft'
        builder.dft.metadata['options']['resources']['num_machines'] = 4
        builder.dft.metadata['options']['max_wallclock_seconds'] = 60*60
        builder.dft.structure = structure
        builder.dft.parameters = Dict(dict={
            'restart' : True,
            'driver' : {
                'maxiter' : 500
            },
            'cosmo' : {
                'dielec' : 78.0,
                'rsolv' : 0.50
            },
            'dft' : {
                'iterations' : 500,
                'xc' : 'xbnl07 0.90 lyp 1.00 hfexch 1.00',
                'cam 0.33 cam_alpha 0.0 cam_beta 1.0' : '',
                'direct' : '',
                'vectors' : 'input uhf.movecs output dft.movecs',
                'mult' : spin_mult,
                'mulliken' : ''
            },
            'task' : 'dft energy'
        })   

        # TDDFT parameters
        builder.tddft.code = code
        builder.tddft.metadata = copy.deepcopy(metadata)
        builder.tddft.metadata['call_link_label'] = 'tddft'
        builder.tddft.metadata['options']['resources']['num_machines'] = 4
        builder.tddft.metadata['options']['max_wallclock_seconds'] = 2*60*60
        builder.tddft.structure = structure
        builder.tddft.parameters = Dict(dict={
            'restart' : True,
            'dft' : {
                'iterations' : 500,
                'xc' : 'xbnl07 0.90 lyp 1.00 hfexch 1.00',
                'cam 0.33 cam_alpha 0.0 cam_beta 1.0' : '',
                'direct' : '',
                'vectors' : 'input dft.movecs',
                'mult' : spin_mult,
                'mulliken' : ''
            },
            'tddft' : {
                'cis' : '',
                'NOSINGLET' : '',
                'nroots' : 20,
                'maxiter' : 1000,
                'freeze' : 17
            },
            'task' : 'tddft energy'
        })   

        return builder

    def get_cage_charge(charge,cage_style):
        
        if cage_style == 'octahedra':
            return charge - 6

    def get_point_charges(cage_style,position):

        point_charges = []

        if cage_style == 'octahedra':
            
            point_charges.append((np.array(position) + np.array([2,0,0])).tolist() + [-1])
            point_charges.append((np.array(position) + np.array([0,2,0])).tolist() + [-1])
            point_charges.append((np.array(position) + np.array([0,0,2])).tolist() + [-1])
            point_charges.append((np.array(position) + np.array([-2,0,0])).tolist() + [-1])
            point_charges.append((np.array(position) + np.array([0,-2,0])).tolist() + [-1])
            point_charges.append((np.array(position) + np.array([0,0,-2])).tolist() + [-1])

        return point_charges

    def run_cage(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'cage'))
        parameters = self.inputs.cage.parameters.get_dict()
        inputs.metadata = self.inputs.cage.metadata
        inputs.code = self.inputs.cage.code
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.cage.structure

        future_cage = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching cage NwchemCalculation <{future_cage.pk}>')
        return ToContext(calc_cage=future_cage)

    def check_cage(self):

        calculation = self.ctx.calc_cage

        if not calculation.is_finished_ok:
            self.report(f'cage NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.CAGE_FAILED

        self.ctx.calc_parent_folder = calculation.outputs.remote_folder

    def run_ligand(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'ligand'))
        inputs.parent_folder = self.ctx.calc_parent_folder
        parameters = self.inputs.ligand.parameters.get_dict()
        inputs.metadata = self.inputs.ligand.metadata
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.ligand.structure

        future_ligand = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching ligand NwchemCalculation <{future_ligand.pk}>')
        return ToContext(calc_ligand=future_ligand)

    def check_ligand(self):

        calculation = self.ctx.calc_ligand

        if not calculation.is_finished_ok:
            self.report(f'ligand NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.LIGAND_FAILED

        self.ctx.calc_parent_folder = calculation.outputs.remote_folder

    def run_full(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'full'))
        inputs.parent_folder = self.ctx.calc_parent_folder
        parameters = self.inputs.full.parameters.get_dict()
        inputs.metadata = self.inputs.full.metadata
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.full.structure

        future_full = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching cage NwchemCalculation <{future_full.pk}>')
        return ToContext(calc_full=future_full)

    def check_full(self):

        calculation = self.ctx.calc_full

        if not calculation.is_finished_ok:
            self.report(f'full NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.FULL_FAILED

        self.ctx.calc_parent_folder = calculation.outputs.remote_folder

    def run_uhf(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'uhf'))
        inputs.parent_folder = self.ctx.calc_parent_folder
        parameters = self.inputs.uhf.parameters.get_dict()
        inputs.metadata = self.inputs.uhf.metadata
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.uhf.structure

        future_uhf = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching uhf NwchemCalculation <{future_uhf.pk}>')
        return ToContext(calc_uhf=future_uhf)

    def check_uhf(self):

        calculation = self.ctx.calc_uhf

        if not calculation.is_finished_ok:
            self.report(f'uhf NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.UHF_FAILED

        self.ctx.calc_parent_folder = calculation.outputs.remote_folder

    def run_dft(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'dft'))
        inputs.parent_folder = self.ctx.calc_parent_folder
        parameters = self.inputs.dft.parameters.get_dict()
        inputs.metadata = self.inputs.dft.metadata
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.dft.structure

        future_dft = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching dft NwchemCalculation <{future_dft.pk}>')
        return ToContext(calc_dft=future_dft)
 
    def check_dft(self):

        calculation = self.ctx.calc_dft

        if not calculation.is_finished_ok:
            self.report(f'dft NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.DFT_FAILED

        self.ctx.calc_parent_folder = calculation.outputs.remote_folder

    def run_tddft(self):

        inputs = AttributeDict(self.exposed_inputs(NwchemCalculation,'tddft'))
        inputs.parent_folder = self.ctx.calc_parent_folder
        parameters = self.inputs.tddft.parameters.get_dict()
        inputs.metadata = self.inputs.tddft.metadata
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.inputs.dft.structure

        future_tddft = self.submit(NwchemCalculation,**inputs)
        self.report(f'launching tddft NwchemCalculation <{future_tddft.pk}>')
        return ToContext(calc_tddft=future_tddft)

    def check_tddft(self):

        calculation = self.ctx.calc_tddft

        if not calculation.is_finished_ok:
            self.report(f'tddft NwchemCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.TDDFT_FAILED

    def results(self):

        calculation = self.ctx.calc_tddft
        if calculation.is_finished_ok:
            self.report(f'workchain finished')

        self.out_many(self.exposed_outputs(calculation, NwchemCalculation))

    def cu_basis():
        basis = '''   S
      5.430321E+06           7.801026E-06          -4.404706E-06           9.704682E-07          -1.959354E-07          -3.532229E-07     
      8.131665E+05           6.065666E-05          -3.424801E-05           7.549245E-06          -1.523472E-06          -2.798812E-06     
      1.850544E+05           3.188964E-04          -1.801238E-04           3.968892E-05          -8.014808E-06          -1.432517E-05     
      5.241466E+04           1.344687E-03          -7.600455E-04           1.677200E-04          -3.383992E-05          -6.270946E-05     
      1.709868E+04           4.869050E-03          -2.759348E-03           6.095101E-04          -1.231191E-04          -2.179490E-04     
      6.171994E+03           1.561013E-02          -8.900970E-03           1.978846E-03          -3.992085E-04          -7.474316E-04     
      2.406481E+03           4.452077E-02          -2.579378E-02           5.798049E-03          -1.171900E-03          -2.049271E-03     
      9.972584E+02           1.103111E-01          -6.623861E-02           1.534158E-02          -3.096141E-03          -5.885203E-03     
      4.339289E+02           2.220342E-01          -1.445927E-01           3.540484E-02          -7.171993E-03          -1.226885E-02     
      1.962869E+02           3.133739E-01          -2.440110E-01           6.702098E-02          -1.356621E-02          -2.683147E-02     
      9.104280E+01           2.315121E-01          -2.504837E-01           8.026945E-02          -1.643989E-02          -2.479261E-02     
      4.138425E+01           7.640920E-02           2.852577E-02          -1.927231E-02           4.107628E-03          -5.984746E-03     
      1.993278E+01           1.103818E-01           5.115874E-01          -3.160129E-01           6.693964E-02           1.557124E-01     
      9.581891E+00           1.094372E-01           4.928061E-01          -4.573162E-01           1.028221E-01           1.436683E-01     
      4.234516E+00           1.836311E-02           8.788437E-02           1.550841E-01          -4.422945E-02           8.374103E-03     
      1.985814E+00          -6.043084E-04          -5.820281E-03           7.202872E-01          -2.031191E-01          -7.460711E-01     
      8.670830E-01           5.092245E-05           2.013508E-04           3.885122E-01          -2.230022E-01           1.244367E-01     
      1.813390E-01          -5.540730E-05          -5.182553E-04           1.924326E-02           2.517975E-01           1.510110E+00     
      8.365700E-02           3.969482E-05           3.731503E-04          -7.103807E-03           5.650091E-01          -3.477122E-01     
Cu    S
      3.626700E-02           1.0000000        
Cu    S
      1.572000E-02           1.0000000        
Cu    P
      2.276057E+04           4.000000E-05          -1.500000E-05           3.000000E-06           5.000000E-06     
      5.387679E+03           3.610000E-04          -1.310000E-04           2.500000E-05           4.900000E-05     
      1.749945E+03           2.083000E-03          -7.550000E-04           1.470000E-04           2.780000E-04     
      6.696653E+02           9.197000E-03          -3.359000E-03           6.560000E-04           1.253000E-03     
      2.841948E+02           3.266000E-02          -1.208100E-02           2.351000E-03           4.447000E-03     
      1.296077E+02           9.379500E-02          -3.570300E-02           7.004000E-03           1.337000E-02     
      6.225415E+01           2.082740E-01          -8.250200E-02           1.613100E-02           3.046900E-02     
      3.092964E+01           3.339930E-01          -1.398900E-01           2.777000E-02           5.344700E-02     
      1.575827E+01           3.324930E-01          -1.407290E-01           2.756700E-02           5.263900E-02     
      8.094211E+00           1.547280E-01           3.876600E-02          -1.011500E-02          -1.688100E-02     
      4.046921E+00           2.127100E-02           3.426950E-01          -8.100900E-02          -1.794480E-01     
      1.967869E+00          -1.690000E-03           4.523100E-01          -1.104090E-01          -2.095880E-01     
      9.252950E-01          -1.516000E-03           2.770540E-01          -7.173200E-02          -3.963300E-02     
      3.529920E-01          -2.420000E-04           4.388500E-02           1.879300E-01           5.021300E-01     
      1.273070E-01           2.300000E-05          -2.802000E-03           5.646290E-01           5.811110E-01     
Cu    P
      4.435600E-02           1.0000000        
Cu    P
      1.545000E-02           1.0000000        
Cu    D
      1.738970E+02           2.700000E-03          -3.363000E-03     
      5.188690E+01           2.090900E-02          -2.607900E-02     
      1.934190E+01           8.440800E-02          -1.082310E-01     
      7.975720E+00           2.139990E-01          -2.822170E-01     
      3.398230E+00           3.359800E-01          -3.471900E-01     
      1.409320E+00           3.573010E-01           2.671100E-02     
      5.488580E-01           2.645780E-01           4.920470E-01     
Cu    D
      1.901990E-01           1.0000000        
Cu    D
      6.591000E-02           1.0000000        
Cu    F
      5.028600E+00           4.242800E-01     
      1.259400E+00           7.630250E-01     
Cu    F
      4.617200E-01           1.0000000        '''
        return basis
