import sourcedefender
from .session import Session
import unittest


class TestRLXMOOCAPI(unittest.TestCase):
    
    def setUp(self):
        self.assertTrue("endpoint" in dir(self), "must set API endpoint, use for instance \n\n    TestRLXMOOCAPI.endpoint='http//:localhost:8000'\n\nbefore running the tests")
        self.assertTrue("adminpwd" in dir(self), "must set amdin pwd, use for instance \n\n    TestRLXMOOCAPI.adminpwd='1234'\n\nbefore running the tests")
        
        self.admin_session = Session(self.endpoint).login("admin", self.adminpwd)
        if self.admin_session.user_exists("rramosp"):
            self.admin_session.delete_user("rramosp")
        self.admin_session.create_user("rramosp", "hola", "raul ramos")
        if self.admin_session.user_exists("pico"):
            self.admin_session.delete_user("pico")
        self.admin_session.create_user("pico", "hola", "pico nada")
        if self.admin_session.user_exists("learner"):
            self.admin_session.delete_user("learner")
        self.admin_session.create_user("learner", "hola", "any student")
        
        self.user1_session = Session(self.endpoint).login("rramosp", "hola")
        self.user2_session = Session(self.endpoint).login("pico", "hola")
        self.user3_session = Session(self.endpoint).login("learner", "hola")
        
        mean_grade = """
        def tasks_grades_aggregation_fn(x):
            import numpy as np
            return np.mean(x.values())
        """

        self.course1_spec = {
                "course_id": "20191.Logic3",
                "course_description": "Logica III",
                "labs": [
                     {
                        "lab_id": "lab_01",
                        "name": "Lab number 1", 
                        "description": "binary search trees",
                        "start_week": 2, 
                        "weeks_duration": 1,
                        "tasks_grades_aggregation_code": mean_grade,
                        "tasks": [
                            {
                                "task_id": "task_01",
                                "name": "Task number 01"
                            },
                            {
                                "task_id": "task_02",
                                "name": "Task number 02"
                            }            
                        ],
                     },
                     {  "lab_id": "lab_02",
                        "name": "Lab number 1", 
                        "description": "binary search trees",
                        "start_week": 3, 
                        "weeks_duration": 1,
                        "tasks_grades_aggregation_code": mean_grade,
                        "tasks": [
                            {
                                "task_id": "task_A",
                                "name": "Task catalog A"
                            }                
                        ],
                  }

                ]

            }
        
        self.course2_spec = {
                "course_id": "20193.IA",
                "course_description": "IA for dummies",
                "labs": [
                     {
                        "lab_id": "lab_A",
                        "name": "Lab code A", 
                        "description": "any lab",
                        "start_week": 2, 
                        "weeks_duration": 1,
                        "tasks_grades_aggregation_code": mean_grade,
                        "tasks": [
                            {
                                "task_id": "task_A",
                                "name": "Task number A"
                            }            
                        ],
                     },
                ]
            }

        if self.admin_session.course_exists(self.course1_spec["course_id"]):
            self.admin_session.delete_course(self.course1_spec["course_id"])
        if self.admin_session.course_exists(self.course2_spec["course_id"]):
            self.admin_session.delete_course(self.course2_spec["course_id"])
            
        if self.admin_session.user_course_exists(self.course2_spec["course_id"], self.user3_session.user):
            self.admin_session.delete_user_course(self.course2_spec["course_id"], self.user3_session.user)

        
        self.grader_source = """
def task_grader(cells):
    return "TEST 1: 13" in cells["validation"][0]["outputs"][0]["text"]

        """
        self.sample_submission_correct = \
           { 'validation': [{'metadata': {'trusted': True, 'task_01': 'validation'},
                             'cell_type': 'code',
                             'source': 'print ("TEST 1:", student_function_f(2,3))\nprint ("TEST 2:", student_function_f(-2,10))\n',
                             'outputs': [{'output_type': 'stream',
                                          'text': 'TEST 1: 13\nTEST 2: -10\n',
                                          'name': 'stdout'}]}]}

        self.sample_submission_incorrect = \
           { 'validation': [{'metadata': {'trusted': True, 'task_01': 'validation'},
                             'cell_type': 'code',
                             'source': 'print ("TEST 1:", student_function_f(2,3))\nprint ("TEST 2:", student_function_f(-2,10))\n',
                             'outputs': [{'output_type': 'stream',
                                          'text': 'TEST 1: 34\nTEST 2: -10\n',
                                          'name': 'stdout'}]}]}


        self.grader_function_name = "task_grader"

    def tearDown(self):
        if self.admin_session.course_exists(self.course1_spec["course_id"]):
            self.admin_session.delete_course(self.course1_spec["course_id"])
        if self.admin_session.course_exists(self.course2_spec["course_id"]):
            self.admin_session.delete_course(self.course2_spec["course_id"])
            
        if self.admin_session.user_course_exists(self.course2_spec["course_id"], self.user3_session.user):
            self.admin_session.delete_user_course(self.course2_spec["course_id"], self.user3_session.user)

        
    def testLogin(self):
        self.assertTrue(type(self.admin_session.token)==str, "admin login not ok")
        self.assertTrue(type(self.user1_session.token)==str, "user1 login not ok")
        self.assertTrue(type(self.user2_session.token)==str, "user2 login not ok")
        
    
    def testUserCannotCreateUsers(self):
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user1_session.create_user("rramosp", "hola", "raul ramos")

    def testUserAccessHisOwnInfoButNotOthers(self):
        u1 = self.user1_session.get_user(self.user1_session.user)        
        self.assertEqual(u1["user_id"], self.user1_session.user)
        
        u2 = self.user2_session.get_user(self.user2_session.user)        
        self.assertEqual(u2["user_id"], self.user2_session.user)
        
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user1_session.get_user(self.user2_session.user)        

        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user2_session.get_user(self.user1_session.user)        


    def testCourseCreationAndAccess(self):
        # create a course
        self.admin_session.create_course(self.course1_spec)        

        # check it can be accessed
        c = self.admin_session.get_course(self.course1_spec["course_id"])
        self.assertTrue(c["course_id"]==self.course1_spec["course_id"])

        # check it can be accessed also by any user
        c = self.user1_session.get_course(self.course1_spec["course_id"])
        self.assertTrue(c["course_id"]==self.course1_spec["course_id"])        
        
        # check exception is raised if trying to create existing course
        with self.assertRaisesRegex(ValueError, "already exists") as e:
            self.admin_session.create_course(self.course1_spec)        

        # delete course
        self.admin_session.delete_course(self.course1_spec["course_id"])

        # check does not exist
        with self.assertRaisesRegex(ValueError, "does not exist") as e:
            self.admin_session.get_course(self.course1_spec["course_id"])
        
        # users cannot create courses
        with self.assertRaisesRegex(ValueError, "only admin can add courses") as e:
            self.user1_session.create_course(self.course1_spec)      

    def testDelegateCourseOwnership(self):
        
        # create a course
        self.admin_session.create_course(self.course1_spec, 
                                         owner=self.user1_session.user) 
        
        # user can change course
        c = self.course1_spec.copy()
        c["course_description"] = "DUMMY"
        self.user1_session.update_course(c)
        cc = self.user2_session.get_course(c["course_id"])
        self.assertEqual(cc["course_spec"]["course_description"],c["course_description"])
        
        # but not the other user
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user2_session.update_course(c)

    def testDefineGraders(self):

        # create a course
        self.admin_session.create_course(self.course1_spec, owner=self.user1_session.user) 

        # check only user1 can set grader
        self.user1_session.set_grader(self.course1_spec["course_id"], "lab_01", "task_01", 
                                      self.grader_source, self.grader_function_name)
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user2_session.set_grader(self.course1_spec["course_id"], "lab_01", "task_01", 
                                          self.grader_source, self.grader_function_name)

        # check only user1 can access grader
        g = self.user1_session.get_grader(self.course1_spec["course_id"], "lab_01", "task_01")
    
        self.assertEqual(g["grader_source"], self.grader_source)
        self.assertEqual(g["grader_function_name"], self.grader_function_name)

        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            g = self.user2_session.get_grader(self.course1_spec["course_id"], "lab_01", "task_01")

    def testCreateUserCourse(self):

        # create a course
        self.admin_session.create_course(self.course2_spec, owner=self.user1_session.user) 
        
        # create user course
        self.user1_session.create_user_course(self.course2_spec["course_id"],
                                              self.user3_session.user,
                                              "2019-09-09")
        
        # learner can see the course    
        c = self.user3_session.get_user_course(self.course2_spec["course_id"], self.user3_session.user)
        self.assertEqual(c["course_id"], self.course2_spec["course_id"])
        
        # but not other learners
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user2_session.get_user_course(self.course2_spec["course_id"], self.user3_session.user)

        # has dates start/end
        for k in [c["user_course_spec"].keys(), c["user_course_spec"]["labs"][0].keys()]:
            self.assertTrue("start_date" in k, "no start_date in user course or labs")
            self.assertTrue("end_date"   in k, "no end_date in user course or labs")
        
        # delete courses
        self.user1_session.delete_user_course(self.course2_spec["course_id"], self.user3_session.user)
        self.admin_session.delete_course(self.course2_spec["course_id"])
        
    def testTaskSubmission(self):
        # create a course
        self.admin_session.create_course(self.course2_spec, owner=self.user1_session.user) 
        
        # create user course
        cid = self.course2_spec["course_id"]
        uid = self.user3_session.user
        self.user1_session.create_user_course(cid, uid, "2019-09-09")
        self.user1_session.set_grader(cid, "lab_A", "task_A", self.grader_source, self.grader_function_name)

        # admin, owner or learner himself can submit in him name
        r1 = self.user3_session.submit_task(cid, "lab_A", "task_A", self.sample_submission_correct)
        r2 = self.user1_session.submit_task(cid, "lab_A", "task_A", self.sample_submission_correct, user_id=uid)
        r3 = self.admin_session.submit_task(cid, "lab_A", "task_A", self.sample_submission_correct, user_id=uid)
        self.assertEqual(r1["grade"], 1, "grade must be 1 for correct submission")
        self.assertEqual(r2["grade"], 1, "grade must be 1 for correct submission")
        self.assertEqual(r3["grade"], 1, "grade must be 1 for correct submission")

        r1 = self.user3_session.submit_task(cid, "lab_A", "task_A", self.sample_submission_incorrect)
        self.assertEqual(r1["grade"], 0, "grade must be 0 for incorrect submission")
        
        # but not other learners
        with self.assertRaisesRegex(ValueError, "unauthorized") as e:
            self.user2_session.submit_task(cid, "lab_A", "task_A", 
                                           self.sample_submission_correct, user_id=uid)

            
        # delete courses
        self.user1_session.delete_user_course(self.course2_spec["course_id"], self.user3_session.user)
        self.admin_session.delete_course(self.course2_spec["course_id"])
            
def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRLXMOOCAPI)
    unittest.TextTestRunner().run(suite)
