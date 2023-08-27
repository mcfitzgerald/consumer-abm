import unittest
from cabm_agents import ConsumerAgent


class TestConsumerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ConsumerAgent(1, None)

    def test_set_purchase_behavior(self):
        # Test when pantry_stock <= pantry_min and price has dropped        
        self.agent.pantry_stock = 5
        self.agent.pantry_min = 10
        self.agent.current_price = 10
        self.agent.last_product_price = 20
        self.agent.set_purchase_behavior()
        self.assertEqual(self.agent.purchase_behavior, "buy_maximum")

        # Test when pantry_stock <= pantry_min and price has not dropped
        self.agent.current_price = 30
        self.agent.set_purchase_behavior()
        self.assertEqual(self.agent.purchase_behavior, "buy_minimum")

        # Test when pantry_min < pantry_stock < pantry_max and price has dropped
        self.agent.pantry_stock = 15
        self.agent.pantry_max = 20
        self.agent.current_price = 10
        self.agent.set_purchase_behavior()
        self.assertEqual(self.agent.purchase_behavior, "buy_maximum")

        # Test when pantry_min < pantry_stock < pantry_max and price has not dropped
        self.agent.current_price = 30
        self.agent.set_purchase_behavior()
        self.assertEqual(self.agent.purchase_behavior, "buy_some_or_none")

        # Test when pantry_stock >= pantry_max
        self.agent.pantry_stock = 25
        self.agent.set_purchase_behavior()
        self.assertEqual(self.agent.purchase_behavior, "buy_none")


if __name__ == "__main__":
    unittest.main()
