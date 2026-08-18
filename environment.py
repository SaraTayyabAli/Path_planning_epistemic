import copy
import math
import matplotlib
matplotlib.use('Agg')

class Env:
    def __init__(self, agent_start, agent_goals, obstacles, size=(10,10)):
        self.unit = 20
        self.pixel = self.unit * 0.5
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.size = size
        self.obstacles = copy.deepcopy(obstacles)
        self.height = size[0]
        self.width = size[1]
        self.start = agent_start
        self.goal = [agent_goals]
        self.able_state = self.traversable_state()
    
    def reset(self):
        """Reset to initial start point"""
        state = self.start
        return state

    def step(self,state,action,tag='train'):
        next_state = self.get_state_action_space(state,action)
        if tag == 'train':
            if next_state in self.goal:
                reward = 10
            else:
                reward = -0.1
        else:
            if next_state in self.goal:
                reward = 10
            else:
                reward = -0.1   
        return next_state, reward
    
    def get_state_action_space(self,state,action):
        next_state = tuple()
        if action == 0:  # up
            next_state = (state[0], state[1]-1)
        if action == 1:  # down
            next_state = (state[0], state[1]+1)
        if action == 2:  # left
            next_state = (state[0]-1, state[1])
        if action== 3:  # right
            next_state = (state[0]+1, state[1])
        return next_state
    
    # enabled state
    def traversable_state(self):
        able_state = set()
        for i in range(self.width):
            for j in range(self.height):
                able_state.add((i,j))
        able_state = able_state - self.obstacles
        return able_state  

    def create_grid_map(self, filename, path=None, rows=10, cols=10, cell_size=20, margin=20):
        """
        Create grid map PDF file (with path drawing)
        Parameters:
        filename - Output filename
        path     - Path coordinate list [(x0,y0), (x1,y1)...]
        rows     - Number of rows
        cols     - Number of columns
        cell_size- Cell size (in points)
        margin   - Page margin
        """
        # Initialize path parameters
        from reportlab.pdfgen import canvas
        path = path or []
        
        # Calculate page dimensions
        page_width = cols * cell_size + 2 * margin
        page_height = rows * cell_size + 2 * margin
        
        # Create PDF canvas
        c = canvas.Canvas(filename, pagesize=(page_width, page_height))
        
        # Draw grid lines
        for i in range(rows + 1):
            y = margin + i * cell_size
            c.line(margin, y, page_width - margin, y)
            
        for j in range(cols + 1):
            x = margin + j * cell_size
            c.line(x, margin, x, page_height - margin)
        
        # Draw obstacles
        for (x, y) in self.obstacles:
            pdf_y = rows - y - 1
            c.rect(
                margin + x * cell_size,
                margin + pdf_y * cell_size,
                cell_size, cell_size,
                fill=1,
                stroke=0
            )

        # Draw path (new feature)
        if len(path) >= 2:
            # Set path style
            c.setStrokeColorRGB(0, 0, 1)  # Blue path
            c.setLineWidth(3)             # 3 point line width
            
            # Draw path line segments
            for i in range(len(path)-1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                # Coordinate system conversion
                pdf_y1 = rows - y1 - 1
                pdf_y2 = rows - y2 - 1
                
                # Calculate line segment endpoint coordinates
                start_x = margin + x1 * cell_size + cell_size/2
                start_y = margin + pdf_y1 * cell_size + cell_size/2
                end_x = margin + x2 * cell_size + cell_size/2
                end_y = margin + pdf_y2 * cell_size + cell_size/2
                
                c.line(start_x, start_y, end_x, end_y)
            
            # Draw path endpoint markers
            self._draw_path_markers(c, path, rows, cols, cell_size, margin)
        c.showPage()
        c.save()

    def _draw_path_markers(self, canvas, path, rows, cols, cell_size, margin):
            """Draw path endpoint markers (internal method)"""
            # Start point marker (green)
            start_x, start_y = path[0]
            pdf_y = rows - start_y - 1
            canvas.setFillColorRGB(0, 1, 0)
            canvas.circle(
                margin + start_x * cell_size + cell_size/2,
                margin + pdf_y * cell_size + cell_size/2,
                5,  # Radius 5 points
                fill=1
            )
            
            # End point marker (red)
            end_x, end_y = path[-1]
            pdf_y = rows - end_y - 1
            canvas.setFillColorRGB(1, 0, 0)
            canvas.circle(
                margin + end_x * cell_size + cell_size/2,
                margin + pdf_y * cell_size + cell_size/2,
                5,
                fill=1
            )
    
    # get the neighbor state of the current state
    def get_legal_neighbors_human(self,state):
        
        neighbors = [state]
        #right action
        n = (state[0]+1,state[1])
        if (state[0]+1,state[1]) in self.able_state:
            neighbors.append(n)

        # left action
        n = (state[0]-1,state[1])
        if (state[0]-1,state[1]) in self.able_state:
            neighbors.append(n)

        # down action
        n = (state[0],state[1]+1)
        if (state[0],state[1]+1) in self.able_state:
            neighbors.append(n)

        # up action
        n = (state[0],state[1]-1)
        if (state[0],state[1]-1) in self.able_state:
            neighbors.append(n)
        return neighbors
       
    @staticmethod
    def Eou(state1,state2):
        return math.sqrt(math.pow(state1[0]-state2[0],2)+math.pow(state1[1]-state2[1],2))
    
    @staticmethod
    def manhattan_distance(state1, state2):
        return abs(state1[0]-state2[0]) + abs(state1[1]-state2[1])


