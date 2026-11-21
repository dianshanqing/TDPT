import unittest


class IQAEResourceTests(unittest.TestCase):
    def test_counts_include_inverse_both_reflections_and_measurement(self):
        from rare_precision import iqae_component_cost
        c={key:{'gates':g,'cx':x,'depth':d,'qubits':5} for key,g,x,d in
           [('Atilde',10,2,7),('Atilde_inverse',11,3,8),('S0',12,4,9),('Sf',1,0,1)]}
        r=iqae_component_cost(c,[{'power':0,'shots':2},{'power':3,'shots':4}])
        self.assertEqual(r['A_calls'],18)
        self.assertEqual(r['A_inverse_calls'],12)
        self.assertEqual(r['queries_A_and_inverse'],30)
        self.assertEqual(r['zero_reflection_calls'],12)
        self.assertEqual(r['objective_reflection_calls'],12)
        self.assertEqual(r['gates_excluding_measurement'],18*10+12*(11+12+1))
        self.assertEqual(r['measurement_calls'],6)
        self.assertEqual(r['max_single_circuit_depth_upper_bound'],7+3*(7+8+9+1)+1)


if __name__ == '__main__': unittest.main()
