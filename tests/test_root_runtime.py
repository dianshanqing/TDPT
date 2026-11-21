"""Root-only deployment contracts; detailed copy acceptance uses system temp."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

# Canonical digests captured before migration; these are assertions, not configurations.
PRE_MIGRATION_DIGESTS = {'absorption_AB': '9a87a7dc7b5e5872f9383e3cf31b535ad156f7b272f5af9ad5a98e9dc8b734bb',
 'absorption_BA': '7c346e52a51cecad4d9ab71812b66f14241c892488c1e4090ce46959a0192a8a',
 'compact_abs_n4_tau14': '4ea071a89973c2d74daf044553b23572dff5a6bd1524a585e9a21599a5c9374a',
 'compact_abs_n4_tau19': '214636dfdc0dfb13debaa3435f76e2e71e10559257d1586cf7325fb690809659',
 'compact_abs_n4_tau24': 'fb989ba7e1f4b116f92f0ed5603aede0411157ef69bcf144f8a86b2797cdb5f4',
 'compact_n4_w0003_tau14': '4cc9e92c9943a05606cde0298b887b8ebe6d1c7ec05dcab12cce5e46aef8f659',
 'compact_n4_w0003_tau14_h2': '509bccf99a9517c79c0e8765bfe852ddbd50adb84a4a2789b94eee276c523820',
 'compact_n4_w0003_tau14_h4': 'cdc7a800b389038b64e6d621ff0873c359b736032ce5ec65a1e7369161c333a5',
 'compact_n4_w0003_tau19': '736f3ef520487801502f385559699157cc4f0e0667d5568ce3965e4b3711247f',
 'compact_n4_w0003_tau19_h2': '509c528c9ab13284753393bfd52993cfd196f4540e1c421346674eb04e673ae0',
 'compact_n4_w0003_tau19_h4': 'e36e3855c4e6f0fcf8244fcf1ff366b0808a91355c9b4a5f80fe3352a3290d66',
 'compact_n4_w0003_tau24': 'b47eeb41ef27a0209a86dc839bf9389a36769e0236617ca41f83641ed89de0c9',
 'compact_n4_w0003_tau24_h2': '9de203149377c31a41e7c6d2e0878daed9ab603b50aeefaf4977262cc7c1d150',
 'compact_n4_w0003_tau24_h4': '78e277e4ecbc31a583c59c8b526f889ae86ac34c024a9bac05daffc6eab0e86d',
 'compact_n4_w001_tau14': 'f4328844c88f70988c38c5eb601d3ff3c41330b7ac2974fb8c77126bf2a7e24c',
 'compact_n4_w001_tau19': 'be119107c3504c083c1383390de0bfec641e75ef1adf3b1f278b62fa2fc61cc9',
 'compact_n4_w001_tau24': '115e5e7ce88b10674c14905d2b9476ebcebab3c018afe160dbb36f97c1e66305',
 'compact_n4_w01_tau14': 'b942f7429a5a62c404d15aa76f25e9c127f4c5005bbed20de913baf4f900c351',
 'compact_n4_w01_tau19': 'c8a57f70a1cf588cc424910a733d6cc318eb61e80ec832d07877122c33cd7b75',
 'compact_n4_w01_tau24': 'b1529b541cc003e4cbfb1485ecce9d51425fed8f7bd6ba7d8f6750c453db684c',
 'compact_n8_w0002_tau14': 'd8cf65634695d8457aa69ac65e2830e9f6bd1e48f349305032ec35425c72eedf',
 'compact_n8_w0002_tau14_h2': '5fd5eecb9043712979ac7e1c98ea6eb75df66700a1a2cdfd91e6e2696174ff66',
 'compact_n8_w0002_tau14_h4': 'e30965a5dce048495fa5eed1116ce4b04d61b9ea247813642e27e553ab6005a2',
 'compact_n8_w0002_tau19': 'd8a79a17b2afc13b2ebf77c30819d1d115ed47ab4afce91f59f5597c63cbf157',
 'compact_n8_w0002_tau19_h2': 'af8a801bedf81293d7d962634b4dece0d62608e6dde078cd65647b74e9ccdfc8',
 'compact_n8_w0002_tau19_h4': '0a50d82efb59b0c4327326b707e95656455dfd54021ab947ef5b36c89f0133c1',
 'compact_n8_w0002_tau23': '84ba5f0567e25db17f1a7d6961c47941ee2f020ef486c1f49d0b56903fc14cc1',
 'compact_n8_w0002_tau23_h2': '96e5debc2b278a01810eb45175ea5c3c84696e1f3d25281600226effc26253b0',
 'compact_n8_w0002_tau23_h4': '6ddd34d9770c1a79e51195c5d30e5a92eeebe0b61f7c32ddc85f19929d60d3b0',
 'compact_n8_w0003_tau14': '7c92049c65796a9938f99c94ad8255c9efc8abc7de2f67825292903889c06ab5',
 'compact_n8_w0003_tau14_h2': '02d7f6a7d9ada667829cad52e907498048abe4d6ed4f351bbcb1c2d702f0e05d',
 'compact_n8_w0003_tau14_h4': '3dc6fcbd71252bad6292ec3e9446113c0c09c8d570be958ebc99a7a5297e1422',
 'compact_n8_w0003_tau19': '2d76b4e2dea3b0c361cdbc94ffa4f8aee9053d9ec42fa0fcd24e98514b4ebaa1',
 'compact_n8_w0003_tau19_h2': 'afb415dbe57c823091d4ad8c92308195d3780ce70266f9a38615323a31495ae6',
 'compact_n8_w0003_tau19_h4': '98d455cf0a5e7856ccaad73dff83dd797c74b9e9a7cef83db1e6fb838277486e',
 'compact_n8_w0003_tau23': '3f574cbff78f9c0d69f84ce22c8aeceeb2b27288b88133c1be38a0804972acd7',
 'compact_n8_w0003_tau23_h2': '0f78802b830d0f880f8ff9ed8d651cde9dbc5b7cf968f3671a63c1db5337db1a',
 'compact_n8_w0003_tau23_h4': 'af5de2976c8953f0c5b87bb0eeea7bc45b010b00dce2cbf405c668038a7f207e',
 'compact_n8_w0003_tau24': '97c6ca1017311360c12cc80acc5b7066201b67d5adb372accd9fa0b9480ea480',
 'compact_n8_w0003_tau24_h2': 'c5816157f8ab6541d8121147a7160f7e7d0b1083391fae2ba7f7c83415da9018',
 'compact_n8_w0003_tau24_h4': '85279c83a9ba257ce363175f1930ae413a2353e3eb30cea542ddc969a6af88cf',
 'compact_n8_w001_tau14': 'ee08942f56a2cc1ac1f804653d6c43eb18cf594e222471ffaa1853bda61109ea',
 'compact_n8_w001_tau19': '3d5fe969db31dfae981c8560d440633fce40b6be04de92859fb4596efd5c132e',
 'compact_n8_w001_tau24': 'ad98d140b891fd792d4188e1de552d13cb529d6cbc46067560091d403474d9d8',
 'compact_n8_w01_tau14': '3e503220353db12d43c9abc399031e1e0092e4d36075ec7bca9793df4e7ff986',
 'compact_n8_w01_tau19': '0fa81d3da23947080825b4e8d52bcc8747580a86d191ddb4990b1444f2ed69fd',
 'compact_n8_w01_tau24': '3ddf40b0e6ce535de0cd88fee4a8e330fe4c9ce33cc0d16ba8d28125c70ddf8b',
 'layered_AB_tau2': '666f01818f5d70a2b342c635ba918ac3086c28d95df2e508bf7db51773bc3840',
 'layered_AB_tau2_hhalf': '5a3d752be5284dc703908a34b26d526cfca5e9ddc0b13156fb6715bf8f48b193',
 'layered_AB_tau4': '6db33a10038d6af9188e85a30046518b7ac416ffe2415511c46b7ec22eb1f25b',
 'layered_BA_tau2': 'abf6d06a902353cfc3081f74eb96ca5284a2398171a61bbcecf1cc2371063783',
 'layered_BA_tau4': '66a865b9f84276e062e938f7c8ef656462b0504b4bb296555ef95adcab6b322e',
 'legacy_duct': 'f27fd7709de04e9596f645240a8e41f05829b5d50624826f031c0a0929ab55c2',
 'q_boundary_overlap': 'cf8aa6f5f5897a3188d4f569f813a78e32fdbdad2dd614d2eafb4e611c126f95',
 'q_layered_small': 'f50ffb7fdbd65dc5e1fd1f0ef28e0c50eb3c003ef4e3718708364c549867a9d7',
 'q_legacy_small': 'e2cbc981b88481a89ff5e13886791d5bbf0d7de498ac8cac3ada1db59d7b2cb4',
 'q_mps_one_cell': 'bb5ff28b67c3d8e2f547d537ebd3d33f828c01068e6d1693427b958889e92d0d',
 'q_pure_absorb': '20b7db58792e063b01f88b96f5983b9eed55e022d54ce562150ed659b9588e24',
 'q_pure_scatter': '1fbff88b35fc0c11cbb4770e2cdceb43301eaca43f55e850b12903323965c0e1',
 'q_uniform_small': '17c53186005a03db7107715309cd4ab95b6eefc20115f9112007a1f9620b053f',
 'q_vacuum': '98df16c4f3e4d5008266ef2137df78907cbb992594a9ebba4c35371ea1edceea',
 'uniform_matched_tau2': 'cf3c7443273a0e89990c0d89e30413b89ec19048359696c23d833ef0fe45e795',
 'uniform_matched_tau4': 'eaaf5947f5312281a51b1552e8a5f5de7bf52c643e93b02931366a151a4da59e',
 'uniform_tau1_w5': '98e948e6cb4a23cfb48c5805fdd511cbf40c7fdfa91adc92553c5e43b4969c11',
 'uniform_tau1_w9': '0ed4052aa848389152974594a8c3221ecb580bacd47292c554aa837a017e21b5',
 'uniform_tau2_w5': '5336bbc4ec87ae794caac9bf44fe9057db5b68f8a6b037661db589d74d47826c',
 'uniform_tau2_w5_hhalf': '42be2e82de91ada95a33c1e08932dd54a19b3abe92a5dccd20a12b434cadd293',
 'uniform_tau2_w5_wide': '50a2812b120558f8cd65f81745ce2d5f5c04c4145d41bc3371acb0f052b56ec6',
 'uniform_tau2_w9': '53fece4b45cda5593fa6a4b251bc271ab8df72be3a5a1ca2a559f6c42eee0a15',
 'uniform_tau4_w5': '9137c934e196eebc0f5fc6a38f05150c674c7a047e659961203a76134f069811',
 'uniform_tau4_w9': 'd5517e052998e7d295dd9a0fa4ed5f86ac77f9eaac655b50a065c25615285459',
 'compact_cloud_tau14_b128': '50f714e12418e9125265d5653508c9d9032497161ccf7b869f3d78bc047dc2ec',
 'compact_cloud_tau19_b128': 'a283f6b722a9f612d40b598e6ac5aadf58041504f4f338704b819c7154ac0fc8',
 'compact_cloud_tau23_b128': '54922dc2a269366512539a10c8a7ae1634a1cfb21b418a08b095a00d57a4f64f',
 'compact_tau23_probe': '2051a891a908cde80eace485f9226e46b5e67f8d01e2fa805cd562e6637173ba',
 'compact_quantum_absorption': 'ca9dad792a223656f083ac12e165892f5f6c6d90d84240a9a60dd231f78829c6',
 'compact_quantum_absorption_bounded': '68044c14b9f92e4ae525e58de94a01250e71a5bf5d017a2e942b5467fdefd74f',
 'compact_quantum_absorption_mps': '0da127ca836792007894faf28b57b68e5cc0fa71c93a708a08c992005ee87a2c',
 'compact_quantum_precision': '585f25919e051b03b8b3a9ee8a7627302b450100cc25016587a0c364ad2e9370',
 'compact_quantum_precision_final': 'd60474222644f6e94e8c2f368bd8da44a598dc1a50d261ffa3eed31601e0b47b',
 'compact_quantum_scatter_tau14': 'f2f1e9d2c6d8d896af6e0cb63a34ce172d1834bc237d6a1f71ea2b9102415d8b',
 'compact_quantum_scatter_w0002_tau14': 'ee5ddbd779a9f46b93c7f26327610cf7d82467e7132a9baed7e5cfa610a8f1d8',
 'compact_quantum_scatter_w0002_tau19': '9f99dc67267f5ee79aecdfd1cec438fc688f53345f3c931b624cf9a2b70ac035',
 'compact_quantum_scatter_w0002_tau23': '03bcf51a40f1e70565478a65e4c9e5caf8b16ae812806ada5cb01d8254db4b02',
 'compact_screen': '89b2f37bd0c200b07392ca2d4b59169acef2a500f1115be3470c8ee916dffefd',
 'compact_screen_extension': 'd9ef2a2e46a0697bdbbe8d7e9e311b41ae65815f61c1b86715c2e2f753ae6b80',
 'compact_screen_v1': '2236e4c1082f8fbba7e2197152d3857d858e0b87b91ec3457ef957ada1a99d4f',
 'compact_thread_probe': '8caf19ddb999d7b6651559bca95bbedc109da9ea6262a6eb8d7fd8ff79ac9b38',
 'mps_small': '5f6f8394b25a5984cff5b7ffbec4a7bbed7ab77718eef314795aaf34b3b84e1a',
 'three_models_small': '3fdc7c09f6fb55a7cebdaea1d8ad6c9110a8cafeb11e7e8ea97f97a2dfa56eb3',
 'mps_cloud_layered': '9048d7d3e69f8b1b8530d6c2270f99a1675dd3ac89f0dece035eff9ca35ade8b',
 'mps_cloud_legacy': '14e8d95587dbeaca9d53cf048795fa72faadad6bfabaa10b28365173a17c0614',
 'mps_cloud_uniform': '15cefd3bfd67de2141f28ad68652adacd311f87acda55aea8d90ccefe7d0c509',
 'mps_probe_small': '5cc7d49572ac1ae787b26172d4321f02833b8a5e610993bc696f05373d510804',
 'mps_probe_tight': '294f4b040518bc600d91fd419ddda8ad2c795ba295b6352816bc26e10851244b',
 'screen_small': '3169a0a70d2ac7003dc6b88302e42489b4d88b2fd5652e162a11c42acee20243',
 'svd_calibration': 'b45a3c657984cdb696b8d9827e416a15e6370b95bc6a990937b50f65c468c7e4',
 'validation_small': 'cfe8d80f920f5e984040d1e91a236000891394739f03784e296d1cc92266b84c'}


class RootRuntimeTests(unittest.TestCase):
    def test_all_migrated_config_contents_unchanged(self):
        from presets import get_preset
        for name,expected in PRE_MIGRATION_DIGESTS.items():
            body=json.dumps(get_preset(name),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
            self.assertEqual(hashlib.sha256(body).hexdigest(),expected,name)

    def test_runtime_modules_live_at_root(self):
        for name in ('run_request.py','qc_environment.py','supervised.py','reference_engine.py','rrk_model.py','presets.py','qc_versions.py'):
            self.assertTrue((ROOT/name).is_file(),name)

    def test_preset_and_legacy_alias_resolve_same_model(self):
        from models import load_document, load_model
        a=load_document('compact_n8_w0002_tau23')
        b=load_document('configs/models/compact_deep/compact_n8_w0002_tau23.json')
        self.assertEqual(a,b)
        self.assertEqual(load_model('q_layered_small').model,'layered_slab')
        from run_request import resolve_run_request
        r=resolve_run_request(dict(task='mps_probe',suite='compact_tau23_probe'),project_root=ROOT)
        self.assertEqual(r['backend']['bond_dimensions'],[256])
        self.assertEqual(r['cases'][0]['steps'],15)
        self.assertEqual(r['cases'][0]['model_document'],a)

    def test_versions_export_is_derived(self):
        import qc_versions
        self.assertEqual(qc_versions.requirements(),[k+'=='+v for k,v in qc_versions.PACKAGE_VERSIONS.items()])
        self.assertIn('qiskit-aer==0.17.2',qc_versions.requirements())
        self.assertEqual(qc_versions.PYTHON_VERSION,'3.13.13')

    def test_preset_returns_copy_and_rejects_typo(self):
        from presets import get_preset
        first=get_preset('q_uniform_small'); second=get_preset('q_uniform_small')
        first['geom']['nx']+=1
        self.assertNotEqual(first,second)
        with self.assertRaises(ValueError):get_preset('q_unifrom_small')

    def test_decimal_audit_reads_named_preset_and_legacy_alias(self):
        from compact_classical import audit_precision
        for source in ('compact_abs_n4_tau14',
                       'configs/models/compact_deep/compact_abs_n4_tau14.json'):
            with self.subTest(source=source):
                result=audit_precision([source],{'compact_abs_n4_tau14':9})
                row=result['rows'][0]
                self.assertAlmostEqual(row['p_D_float64']/8.315287191035655e-7,1.,places=12)
                self.assertTrue(row['passes_numeric_target'])
                self.assertEqual(set(row['runs']),{'physical_50','physical_80','fixed_float_50','fixed_float_80'})


if __name__=='__main__':unittest.main()
