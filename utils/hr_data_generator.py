import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

class HRDataGenerator:
    """Generate comprehensive HR sample data for RAG demonstrations"""
    
    def __init__(self):
        self.departments = ["Engineering", "HR", "Finance", "Marketing", "Sales", "Operations", "Legal"]
        self.job_titles = {
            "Engineering": ["Software Engineer", "Senior Engineer", "Tech Lead", "Engineering Manager", "DevOps Engineer"],
            "HR": ["HR Specialist", "HR Manager", "Recruiter", "HR Director", "Benefits Coordinator"],
            "Finance": ["Financial Analyst", "Accountant", "Finance Manager", "CFO", "Controller"],
            "Marketing": ["Marketing Specialist", "Content Manager", "Marketing Director", "Brand Manager"],
            "Sales": ["Sales Representative", "Account Manager", "Sales Director", "Business Development"],
            "Operations": ["Operations Manager", "Project Manager", "Operations Director", "Coordinator"],
            "Legal": ["Legal Counsel", "Paralegal", "Legal Director", "Compliance Officer"]
        }
        
        self.first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", 
                           "James", "Maria", "William", "Jennifer", "Richard", "Patricia", "Charles"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                          "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
        
        self.policy_types = ["Vacation", "Sick Leave", "Remote Work", "Benefits", "Code of Conduct", 
                           "Safety", "Training", "Performance", "Compensation", "Diversity"]
        
        self.document_types = ["Policy", "Procedure", "Form", "Guide", "Manual"]
        
    def generate_comprehensive_data(self, num_employees: int = 50, num_policies: int = 10) -> Dict[str, Any]:
        """Generate comprehensive HR data including employees, policies, and documents"""
        
        employees = self._generate_employees(num_employees)
        policies = self._generate_policies(num_policies)
        documents = self._generate_documents(num_policies * 2)
        
        return {
            "employees": employees,
            "policies": policies,
            "documents": documents,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_employees": len(employees),
                "total_policies": len(policies),
                "total_documents": len(documents)
            }
        }
    
    def generate_graph_data(self, num_employees: int = 30, num_policies: int = 8) -> Dict[str, Any]:
        """Generate data optimized for graph database with relationships"""
        
        employees = self._generate_employees(num_employees)
        policies = self._generate_policies(num_policies)
        
        # Add relationships
        relationships = self._generate_relationships(employees, policies)
        
        return {
            "nodes": {
                "employees": employees,
                "policies": policies,
                "departments": [{"name": dept, "id": f"dept_{dept.lower()}"} for dept in self.departments]
            },
            "relationships": relationships,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "graph_type": "hr_knowledge_graph"
            }
        }
    
    def generate_enterprise_data(self, num_employees: int = 50, num_policies: int = 15) -> Dict[str, Any]:
        """Generate enterprise-grade data with rich metadata for Vertex AI"""
        
        base_data = self.generate_comprehensive_data(num_employees, num_policies)
        
        # Enhance with enterprise features
        base_data["compliance_docs"] = self._generate_compliance_documents()
        base_data["training_materials"] = self._generate_training_materials()
        base_data["org_structure"] = self._generate_org_structure(base_data["employees"])
        
        return base_data
    
    def _generate_employees(self, num_employees: int) -> List[Dict[str, Any]]:
        """Generate employee data"""
        employees = []
        
        for i in range(num_employees):
            department = random.choice(self.departments)
            job_title = random.choice(self.job_titles[department])
            
            employee = {
                "id": f"emp_{i+1:03d}",
                "name": f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
                "email": f"employee{i+1}@company.com",
                "department": department,
                "job_title": job_title,
                "hire_date": (datetime.now() - timedelta(days=random.randint(30, 1825))).isoformat(),
                "salary": random.randint(50000, 150000),
                "manager_id": f"emp_{random.randint(1, max(1, i)):03d}" if i > 0 else None,
                "location": random.choice(["New York", "San Francisco", "Chicago", "Austin", "Remote"]),
                "employment_type": random.choice(["Full-time", "Part-time", "Contract"]),
                "benefits_enrolled": random.choice([True, False]),
                "performance_rating": random.choice(["Exceeds", "Meets", "Below", "Outstanding"]),
                "last_updated": datetime.now().isoformat()
            }
            employees.append(employee)
        
        return employees
    
    def _generate_policies(self, num_policies: int) -> List[Dict[str, Any]]:
        """Generate HR policy documents"""
        policies = []
        
        policy_contents = {
            "Vacation": "Employees are entitled to paid vacation time based on their length of service. New employees receive 10 days annually, increasing to 15 days after 2 years and 20 days after 5 years. Vacation requests must be submitted at least 2 weeks in advance and approved by the direct manager.",
            "Sick Leave": "All employees are provided with sick leave benefits. Full-time employees accrue 8 hours of sick leave per month. Sick leave can be used for personal illness, medical appointments, or caring for immediate family members. A doctor's note may be required for absences exceeding 3 consecutive days.",
            "Remote Work": "The company supports flexible work arrangements including remote work options. Employees may work remotely up to 3 days per week with manager approval. Remote work agreements must be documented and reviewed quarterly. All remote workers must maintain reliable internet and appropriate workspace.",
            "Benefits": "The company provides comprehensive benefits including health insurance, dental, vision, 401(k) with company matching, life insurance, and disability coverage. Benefits enrollment occurs during the annual open enrollment period or within 30 days of hire. Contact HR for detailed benefit information.",
            "Code of Conduct": "All employees must maintain the highest standards of professional conduct. This includes treating colleagues with respect, maintaining confidentiality, avoiding conflicts of interest, and complying with all applicable laws and regulations. Violations may result in disciplinary action up to and including termination.",
            "Safety": "Employee safety is our top priority. All employees must follow safety protocols, report hazards immediately, and participate in required safety training. Personal protective equipment will be provided where necessary. Emergency procedures are posted throughout the facility.",
            "Training": "The company is committed to employee development through various training programs. All employees must complete mandatory compliance training annually. Professional development opportunities are available with manager approval. Training records are maintained by HR.",
            "Performance": "Performance evaluations are conducted annually for all employees. Goals are set at the beginning of each year and progress is reviewed quarterly. Performance ratings directly impact compensation adjustments and career advancement opportunities.",
            "Compensation": "Compensation is based on job responsibilities, performance, market rates, and internal equity. Salary reviews occur annually as part of the performance evaluation process. The company maintains a merit-based pay philosophy with opportunities for advancement.",
            "Diversity": "The company is committed to fostering a diverse and inclusive workplace. We prohibit discrimination based on race, gender, age, religion, sexual orientation, or any other protected characteristic. Diversity training is mandatory for all employees and managers."
        }
        
        for i in range(num_policies):
            policy_type = random.choice(self.policy_types)
            
            policy = {
                "id": f"policy_{i+1:03d}",
                "title": f"{policy_type} Policy",
                "type": policy_type,
                "content": policy_contents.get(policy_type, f"This is the {policy_type.lower()} policy content with detailed guidelines and procedures."),
                "department": random.choice(self.departments + ["All"]),
                "effective_date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                "last_updated": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
                "approval_status": "Approved",
                "priority": random.choice(["High", "Medium", "Low"]),
                "doc_type": "Policy",
                "tags": [policy_type.lower(), "hr", "policy"],
                "applies_to": random.choice(["All Employees", "Full-time Only", "Managers Only"])
            }
            policies.append(policy)
        
        return policies
    
    def _generate_documents(self, num_documents: int) -> List[Dict[str, Any]]:
        """Generate various HR documents"""
        documents = []
        
        document_templates = {
            "Form": [
                "Time Off Request Form - Use this form to request vacation, sick leave, or personal time off.",
                "Expense Reimbursement Form - Submit business expenses for reimbursement using this form.",
                "Performance Review Form - Annual performance evaluation form for employees and managers.",
                "Training Request Form - Request approval for external training or conference attendance.",
                "IT Equipment Request - Request new computer equipment or software licenses."
            ],
            "Guide": [
                "New Employee Onboarding Guide - Complete guide for new hires covering first day procedures, benefits enrollment, and company culture.",
                "Manager's Handbook - Comprehensive guide for managers covering hiring, performance management, and employee relations.",
                "Benefits Enrollment Guide - Step-by-step guide for selecting and enrolling in company benefits.",
                "Remote Work Setup Guide - Instructions for setting up a productive home office environment.",
                "Emergency Procedures Guide - Safety procedures and emergency contact information."
            ],
            "Procedure": [
                "Hiring Procedure - Step-by-step process for recruiting, interviewing, and hiring new employees.",
                "Disciplinary Action Procedure - Guidelines for addressing performance and conduct issues.",
                "Grievance Procedure - Process for employees to file complaints and seek resolution.",
                "Promotion Procedure - Criteria and process for employee promotions and career advancement.",
                "Exit Interview Procedure - Process for conducting exit interviews with departing employees."
            ]
        }
        
        for i in range(num_documents):
            doc_type = random.choice(self.document_types)
            if doc_type in document_templates:
                content = random.choice(document_templates[doc_type])
            else:
                content = f"This is a {doc_type.lower()} document containing important HR information and procedures."
            
            document = {
                "id": f"doc_{i+1:03d}",
                "title": content.split(" - ")[0] if " - " in content else f"{doc_type} Document {i+1}",
                "content": content,
                "doc_type": doc_type,
                "department": random.choice(self.departments + ["All"]),
                "created_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "last_updated": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "author": f"HR Team",
                "priority": random.choice(["High", "Medium", "Low"]),
                "status": "Active",
                "file_type": "text",
                "tags": [doc_type.lower(), "hr", "documentation"]
            }
            documents.append(document)
        
        return documents
    
    def _generate_relationships(self, employees: List[Dict], policies: List[Dict]) -> List[Dict[str, Any]]:
        """Generate relationships for graph database"""
        relationships = []
        
        # Employee-Department relationships
        for employee in employees:
            relationships.append({
                "from": employee["id"],
                "to": f"dept_{employee['department'].lower()}",
                "type": "WORKS_IN",
                "properties": {"since": employee["hire_date"]}
            })
        
        # Manager-Employee relationships
        for employee in employees:
            if employee["manager_id"]:
                relationships.append({
                    "from": employee["manager_id"],
                    "to": employee["id"],
                    "type": "MANAGES",
                    "properties": {"since": employee["hire_date"]}
                })
        
        # Policy-Department relationships
        for policy in policies:
            if policy["department"] != "All":
                relationships.append({
                    "from": policy["id"],
                    "to": f"dept_{policy['department'].lower()}",
                    "type": "APPLIES_TO",
                    "properties": {"effective_date": policy["effective_date"]}
                })
        
        return relationships
    
    def _generate_compliance_documents(self) -> List[Dict[str, Any]]:
        """Generate compliance-related documents"""
        compliance_docs = [
            {
                "id": "comp_001",
                "title": "GDPR Compliance Guide",
                "content": "Guidelines for handling personal data in compliance with GDPR regulations.",
                "type": "Compliance",
                "regulation": "GDPR",
                "mandatory": True
            },
            {
                "id": "comp_002", 
                "title": "SOX Compliance Procedures",
                "content": "Sarbanes-Oxley compliance procedures for financial reporting.",
                "type": "Compliance",
                "regulation": "SOX",
                "mandatory": True
            }
        ]
        return compliance_docs
    
    def _generate_training_materials(self) -> List[Dict[str, Any]]:
        """Generate training materials"""
        training_materials = [
            {
                "id": "train_001",
                "title": "Diversity and Inclusion Training",
                "content": "Comprehensive training on creating an inclusive workplace.",
                "type": "Training",
                "duration_hours": 4,
                "mandatory": True
            },
            {
                "id": "train_002",
                "title": "Cybersecurity Awareness",
                "content": "Training on identifying and preventing cybersecurity threats.",
                "type": "Training", 
                "duration_hours": 2,
                "mandatory": True
            }
        ]
        return training_materials
    
    def _generate_org_structure(self, employees: List[Dict]) -> Dict[str, Any]:
        """Generate organizational structure"""
        structure = {}
        for dept in self.departments:
            dept_employees = [emp for emp in employees if emp["department"] == dept]
            structure[dept] = {
                "head_count": len(dept_employees),
                "manager": next((emp for emp in dept_employees if "Manager" in emp["job_title"]), None),
                "employees": dept_employees
            }
        return structure

