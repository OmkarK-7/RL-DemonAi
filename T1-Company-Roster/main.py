"""
This is where the actual working of the program will happen!
We'll be Querying stuff for testing whether your classes were defined correctly

Whenever something inappropriate happens (like a function returning false in people.py),
raise a Value Error.
"""
from people import * # import everything!

if __name__ == "__main__":  # Equivalent to int main() {} in C++.
    last_input = 99
    while last_input != 0:
        last_input = int(input("Please enter a query number:"))

        if last_input == 1:
            name = input("Name:")
            age = int(input("Age:"))
            position = input("Position:")
            ID = input("ID:")
            city = input("City:")
            branchcodes = input("Branch(es):")
            # How will you conver this to a list, given that
            # the user will always enter a comma separated list of branch codes?
            # eg>   2,5
            branchcodes=[int(x) for x in branchcodes.split(',')]

            salary = input("Salary: ")
            # Create a new Engineer with given details.
            engineer = Engineer(name, age, ID, city, branchcodes, salary) # Change this

            engineer_roster.append(engineer) # Add him to the list! See people.py for definiton
            
        
        elif last_input == 2:
            # Gather input to create a Salesperson
            # Then add them to the roster
            name = input("Name : ")
            age = input("Age: ")
            city = input("City: ")
            branchcodes = input("Branch Codes: ")
            salary = int(input("Salary: "))
            # For seperating the branchcodes
            branchcodes = [int(code) for code in branchcodes.split(',')]
            # Creating sales roster.
            salesperson = Salesman(name, age, city, branchcodes, salary)
            sales_roster.append(salesperson)

        elif last_input == 3:
            ID = int(input("ID: "))
            # Print employee details for the given employee ID that is given. 
            
            found_employee = None
            for employee in engineer_roster + sales_roster:
                if employee.ID == int(ID):
                    found_employee = employee
                    break
            
            if not found_employee: print("No such employee")
            else:
                print(f"Name: {found_employee.name} and Age: {found_employee.age}")
                print(f"City of Work: {found_employee.city}")

                ## Write code here to list the branch names to
                ## which the employee reports as a comma separated list
                ## eg> Branches: Goregaon,Fort
                branch_names = [','.join(map(str, found_Employee.branchcodes))]
                ## ???? what comes here??
                # print(f"Branches: " + ???? )
                print(f"Branches: {branch_name}")                
                print(f"Salary: {found_employee.salary}")

        elif last_input == 4:
            #### NO IF ELSE ZONE ######################################################
            # Change branch to new branch or add a new branch depending on class
            # Inheritance should automatically do this. 
            # There should be no IF-ELSE or ternary operators in this zone
            ID = int(input("Enter Employee ID to change branch: "))
            new_branch = int(input("Enter new branch code: "))
            Employee = next((employee for employee in engineer_roster + sales_roster if employee.ID == ID), None)

            if Employee is not None:
                Employee.change_city(new_city)
            #### NO IF ELSE ZONE ENDS #################################################

        elif last_input == 5:
            ID = int(input("Enter Employee ID to promote: "))
            # promote employee to next position
            Emmployee = next((employee for employee in engineer_roster + sales_roster if employee.ID == ID), None)
            position = input("Enter new position: ")
            if Empolyee is not None:
                check = Employee.promote(position)
                if check:
                    print(f"Employee {ID} has been promoted to {position}.")
                else:
                    print(f"Promotion not valid.")

        elif last_input == 6:
            ID = int(input("Enter Employee ID to give increment: "))
            # Increment salary of employee.
            Employee = next((employee for employee in engineer_roster + sales_roster if employee.ID == ID),None)
            if Employee is not None:
                Employee.increment(salary)
                print(f"Emplyee {ID}, his salary has been incremented by {salary}.")
            else:
                print("Increment can't be processed.")

        
        elif last_input == 7:
            ID = int(input("Enter Employee ID to find superior: "))
            # Print superior of the sales employee.
            Employee = next((employee for employee in engineer_roster + sales_roster if employee.ID == ID),None)
            if Employee is not None:
                superior_ID, superior_name = Employee.find_superior()
                if superior_ID is not None:
                    print(f"The superior of the Emplyee {ID} is --> {superior_name} ({superior_ID}).")
                else:
                    print(f"Employee {ID} has no superior.")
            else:
                print("Employee not found.")
                
        
        elif last_input == 8:
            ID_E = int(input("Enter Employee ID to add superior: "))
            ID_S = int(input("Enter Employee ID of superior: "))
            # Add superior of a sales employee
            ID_E.add_superior(ID_S)

        else:
            raise ValueError("No such query number defined")

            
            

            


            


        






