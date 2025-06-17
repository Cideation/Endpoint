import unittest
import json
import os
from server import app
import tempfile
import shutil

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.temp_dir = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_health_check(self):
        response = self.app.get('/health')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertEqual(data['version'], '1.0.0')

    def test_upload_file_no_file(self):
        response = self.app.post('/upload')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'No file part')

    def test_upload_file_empty(self):
        response = self.app.post('/upload', data={'file': ''})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'No selected file')

    def test_upload_file_invalid_type(self):
        # Create a temporary file with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            response = self.app.post('/upload', data={'file': (f, 'test.txt')})
            data = json.loads(response.data)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(data['error'], 'File type not allowed')

    def test_parse_invalid_data(self):
        response = self.app.post('/parse', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Invalid input data')

    def test_clean_with_ai_invalid_data(self):
        response = self.app.post('/clean_with_ai', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Invalid input data')

    def test_evaluate_and_push_invalid_data(self):
        response = self.app.post('/evaluate_and_push', json=[])
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Invalid input data')

    def test_push_to_neo4j_invalid_data(self):
        response = self.app.post('/push', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Invalid input data')

    def test_upload_and_parse_dxf(self):
        # Create a sample DXF file
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            f.write(b'0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF')
            f.flush()
            
            # Upload the file
            with open(f.name, 'rb') as dxf_file:
                response = self.app.post('/upload', data={'file': (dxf_file, 'test.dxf')})
                data = json.loads(response.data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(data['filename'], 'test.dxf')

            # Parse the file
            response = self.app.post('/parse', json=[{'name': 'test.dxf', 'shape': 'rectangle'}])
            self.assertEqual(response.status_code, 200)

        # Clean up
        os.unlink(f.name)

if __name__ == '__main__':
    unittest.main() 