import numpy as np

class CobotEnv:
    """
    The Cobot environment
    """

    def __init__(self, n_operators: int, robot_execution_time: list, id_operator: int, mu_operators: tuple, std: float = 0.04) -> None:
        """
        Initializes an Cobot environment

        Parameters
        ----------
        n_operators : int
            Number of human operators
        robot_execution_time : list
            List containing the robot task processing time
        id_operator : int
            Id of the operator
        mu_operators : tuple
            Tuple of shape (n_operators x n_tasks) containing the mean values of the normal distribution for each task for each human operator
        std : float
            Standard deviation of the normal distribution used to sample operators' processing time (default 0.04)
        """
        self.std = std
        self.robot_execution_time = np.array(robot_execution_time) # robot tasks execution time
        self.n_operators = n_operators
        self.mu_operators = mu_operators
        assert len(self.mu_operators) == self.n_operators, "Invalid Input!" # 'mu_operators' must have the same length as n_operators

        # Ids of the tasks performable by robot and human operators
        self.robot_task_id = np.array((7,8,9,11,12,13,14,15,16,17,18,19,20))
        self.operator_task_id = np.array((1,2,3,4,5,6,7,8,9,10,12,13,14))

        #set the initial state and sample new processing time of human operators
        self.reset(id_operator)


    def reset(self, id_operator: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment to the initial state and re-sample new processing time of the human operators and returns the initial observation

        Parameters
        ----------
        id_operator : int
            Id of the operator

        Returns
        -------
        np.ndarray, np.ndarray
            Initial state of the environment and sampled execution time of the human operators
        """

        self.operators_sampled_time = self.sample_process_time() #operators processing time: (n_operators x n_tasks)
        self.set_operator(id_operator)
        
        # State
        self.robot_done = np.zeros(len(self.robot_task_id), dtype='int') # 0 = not done, 1 = done
        self.robot_scheduled = np.zeros(len(self.robot_task_id), dtype='int') # 0 = not scheduled, 1 = first, 2 = second, ...
        self.operator_done = np.zeros(len(self.operator_task_id), dtype='int') # 0 = not done, 1 = done
        self.operator_scheduled = np.zeros(len(self.operator_task_id), dtype='int') # 0= not scheduled, 1 = first, 2 = second, ...
        self.operator_execution_time = np.copy(self.operators_sampled_time[self.id_operator]) # operators tasks execution time

        return self.get_state(), self.operators_sampled_time


    def step(self, action: list, new_id_operator: int = None) -> tuple[np.ndarray, float, bool]:
        """
        Simulates scheduling action and calculates rewards

        Parameters
        ----------
        action : list
            List of arrays containing the scheduled tasks for robot and operator
        new_id_operator : Optional[int]
            Id of the new operator, None otherwise
        
        Returns
        -------
        np.ndarray, float, bool
            Current state of the environment, reward, whether the episode is over or not
        """
        
        robot_action, operator_action = action

        assert len(robot_action)==len(self.robot_task_id) and len(operator_action)==len(self.operator_task_id), "Invalid Input!"
        
        robot_action = np.array(robot_action)
        operator_action = np.array(operator_action)

        # change operator and update execution time in state
        if new_id_operator is not None:

            # save previous operator
            old_id_operator = self.id_operator
            # update id operator and execution time
            self.set_operator(new_id_operator)
            # only update the time of those tasks that are not done by robot or operator yet 
            self.operator_execution_time[np.intersect1d(np.where(self.operator_done==0)[0], np.where(self.robot_done==0)[0])] = \
                self.operators_sampled_time[self.id_operator][np.intersect1d(np.where(self.operator_done==0)[0], np.where(self.robot_done==0)[0])]

        # if in the previous step the operator started but not completed a task
        if np.sum(self.operator_done*self.operator_execution_time) < np.sum(self.robot_done*self.robot_execution_time):
            
            # number of the not completed task
            task_number = self.operator_task_id[np.where(self.operator_scheduled==1)[0][0]]

            # restore the previous operator's execution time for the previous task to be completed
            if new_id_operator is not None:
                self.operator_execution_time[np.where(self.operator_scheduled==1)[0][0]] = self.operators_sampled_time[old_id_operator][np.where(self.operator_scheduled==1)[0][0]]
            
            # If the task being processed by the operator has now been scheduled for the robot:
            # 1) remove from the new robot scheduling this task
            # 2) then, add with highest priority that task into the new operator scheduling
            if np.where(self.robot_task_id==task_number)[0].size>0 and robot_action[np.where(self.robot_task_id==task_number)[0][0]] != 0:

                # 1)
                t = robot_action[np.where(self.robot_task_id==task_number)[0][0]]
                robot_action[np.where(self.robot_task_id==task_number)[0][0]] = 0
                robot_action[robot_action>t] -= 1

                # 2)
                operator_action[operator_action>0] += 1
                operator_action[np.where(self.operator_scheduled==1)[0][0]] = 1              
        
            # If the task being processed by the operator has been scheduled for the operator again:
            # give highest priority to this task
            elif operator_action[np.where(self.operator_scheduled==1)[0][0]] != 0:
                
                t = operator_action[np.where(self.operator_scheduled==1)[0][0]]
                operator_action[(operator_action>0) & (operator_action<t)] += 1
                operator_action[np.where(self.operator_scheduled==1)[0][0]] = 1

        # if both Robot and Operator have at least one scheduled task each
        if np.any(robot_action) and np.any(operator_action):
            
            # indeces of the robot's scheduled/non-zero tasks
            ids = np.nonzero(robot_action)[0]
            # index of the robot's highest priority task 
            task_robot = ids[np.argmin(robot_action[ids])]
            
            # update state
            self.robot_done[task_robot] = 1 # set the task as "done"
            robot_action[robot_action != 0] -= 1
            self.robot_scheduled = robot_action
            
            # loop until operator tasks exceed robot step in time or until the operator scheduling is empty
            while np.any(operator_action):

                # indeces of the operator's scheduled/non-zero tasks
                ids = np.nonzero(operator_action)[0]
                
                # index of the operator highest priority task 
                task_operator = ids[np.argmin(operator_action[ids])]
                
                task_duration = self.operator_execution_time[task_operator]

                # if the task finishes before the end of the robot task in time, slip by 1 the operator action and update the operator scheduling in the state
                if np.sum(self.operator_done*self.operator_execution_time) + task_duration < np.sum(self.robot_done*self.robot_execution_time):
                    
                    self.operator_done[task_operator] = 1
                    operator_action[operator_action != 0] -= 1
                    self.operator_scheduled = operator_action

                # if the task finishes exactly as soon as the robot finishes its task, slip by 1 the operator action and then stop the step
                elif np.sum(self.operator_done*self.operator_execution_time) + task_duration == np.sum(self.robot_done*self.robot_execution_time):
                    
                    self.operator_done[task_operator] = 1
                    operator_action[operator_action != 0] -= 1
                    self.operator_scheduled = operator_action    
                    break

                # if the task exceeds robot step in time, just update the operator scheduling in the state and stop the step
                else:
                    
                    self.operator_scheduled = operator_action
                    break
        
        # Operator has no scheduled tasks 
        elif np.any(robot_action):

            # complete all the robot remaining tasks
            ids = np.nonzero(robot_action)[0]
            self.robot_done[ids] = 1
            self.robot_scheduled = np.zeros(len(self.robot_task_id), dtype='int')

        # Robot has no scheduled tasks 
        else:

            # complete all the operator remaining tasks
            ids = np.nonzero(operator_action)[0]
            self.operator_done[ids] = 1
            self.operator_scheduled = np.zeros(len(self.operator_task_id), dtype='int')

        # if the episode is over
        if self.is_over():

            #  return the negative total elapsed time as reward, to minimise completion time
            return self.get_state(), -self.get_total_time(), True
            
        # if the episode is not over
        else:

            # return 0 as reward
            return self.get_state(), 0, False
        

    def sample_process_time(self) -> np.ndarray:
        """
        Samples processing time for each task for each operator

        Returns
        -------
        np.ndarray
            Array of shape (n_operators x n_tasks) containing the processing time for each task for each operator
        """
        
        return np.array([np.around(np.random.normal(loc=self.mu_operators[i], scale=self.std), decimals=3) for i in range(self.n_operators)])


    def set_operator(self, id_operator: int) -> None:
        """
        Set new operator

        Parameters
        ----------
        id_operator : int
            Id of the new operator
        """
        
        assert id_operator < self.n_operators, "Invalid Input!"
        self.id_operator = id_operator


    def is_over(self) -> bool:
        """
        Check whether the episode is over

        Returns
        -------
        bool
            True whether the episode is over, False otherwise
        """
        
        return np.sum(self.robot_done) + np.sum(self.operator_done) == 20.0 # sum of number tasks done by robot and tasks done by operators must be 20 at the end of an episode

    
    def get_total_time(self) -> float:
        """
        Return the total elapsed time of the episode

        Returns
        -------
        float
            Episode elapsed time
        """
        
        # robot total time
        r = np.sum(self.robot_done*self.robot_execution_time)
        # operator total time
        t = np.sum(self.operator_done*self.operator_execution_time)
        
        return max(r,t)


    def get_state(self) -> np.ndarray:
        """
        Return the current state of the environment

        Returns
        -------
        np.ndarray
            Current state
        """
        
        self.state = np.vstack((self.robot_done,
                                self.robot_scheduled,
                                self.robot_execution_time,
                                self.operator_done,
                                self.operator_scheduled,
                                self.operator_execution_time))
        return self.state