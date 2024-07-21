#include <bits/stdc++.h>

using namespace std;

/*BFS模板
void BFS()
{
    queue<node> que;
    node top,son,start,end; //top是queue的最前端元素，son是top所连接到的点，start是起始节点,end是终点

    Q.push(start); //初始状态放进队列
    hash(start) = true; //这里视情况而定，记录是否遍历到的数据结构也可以是visited数组或者是set

    while(!que.empty())
    {
        top = que.front(); //取出队头
        que.pop(); //队头出队，访问完毕

        while(son = top通过没规则能够到达的节点)
        {
            if(son == end) return true; //找到终点！

            if(isValid(son) && !hash(son)) //son是一个合法且没访问过的节点
            {
                que.push(son); //加入队列que
                hash(son) = true; //节点已经访问过
            }
        }
    }
    return false; //没有找到
}
*/

/*
//9.3 T547 省份数量

class Solution1 {//BFS写法
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<bool> visited(n,false);//是否已经访问过
        queue<int> que;//队列，用于BFS
        int top = 0,res = 0;
        for(int i = 0;i < n;i++)
        {
            if(!visited[i])
            {
                res++;
                que.push(i);visited[i] = true;

                while(!que.empty())
                {
                    top = que.front();que.pop();

                    for(int i = 0;i < n;i++)
                    {
                        if(isConnected[top][i] == 1 && !visited[i]) {que.push(i);visited[i] = true;}
                    }
                }
            }
        }
        return res;
    }
};

class Solution2 {//并查集写法
public:
    vector<int> fa;//储存每个节点的父节点
    int Find(int idx)//返回第idx个的父节点
    {
        if(idx = fa[idx]) return idx;
        else
        {
            fa[idx] = find(fa[idx]);
            return fa[idx];
        }
    }
    void Union(int idx1,int idx2)//合并idx1和idx2
    {
        fa[Find(idx1)] = Find(idx2);
    }

    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();fa.resize(n);
        int i = 0;
        for(i = 0;i < n;i++) fa[i] = i;//每个的父节点先是自己

        for(i = 0;i < n;i++)
        {
            for(int j = i + 1;j < n;j++)
            {
                if(isConnected[i][j] == 1)//i,j之间有边
                {
                    Union(i,j);
                }
            }
        }

        int res = 0;
        for(i = 0;i < n;i++)
        {
            if(fa[i] == i) res++;
        }
        return res;
    }
};
*/


/*
//9.3 T802 最终安全状态

class Solution {//拓扑排序
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        int n = graph.size(),i = 0,j = 0;
        vector<int> res;
        stack<int> stk;
        vector<int> Deg(n);//记录每个点的出度
        vector<vector<int>> map(n);//做一个反图，方便查找节点的前驱

        for(i = 0;i < n;i++)
        {
            Deg[i] = graph[i].size();//每个点的出度
            if(Deg[i] == 0) stk.push(i);//出度为0的点入栈

            for(j = 0;j < Deg[i];j++) map[graph[i][j]].push_back(i);//j的前驱是i
        }

        while(!stk.empty())
        {
            i = stk.top();stk.pop();
            res.push_back(i);
            for(int j : map[i])
            {
                if(--Deg[j] == 0) stk.push(j);
            }
        }

        sort(res.begin(),res.end());
        return res;
    }
};

int main()
{
    Solution A;
    vector<vector<int>> graph = {{1,2},{2,3},{5},{0},{5},{},{}};
    vector<int> res = A.eventualSafeNodes(graph);
}
*/


/*
//9.4 T841 钥匙和房间

class Solution {//BFS写法
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int len = rooms.size(),i = 0;
        queue<int> que;
        vector<bool> opened(len,false);int top = 0;

        que.push(0);
        while(!que.empty())
        {
            top = que.front();que.pop();

            if(!opened[top])
            {
                for(int j : rooms[top])
                {
                    if(!opened[j])
                    {
                        que.push(j);
                    }
                }
                opened[top] = true;
            }
        }

        for(i = 0;i < len;i++)
        {
            if(!opened[i]) return false;
        }
        return true;
    }
};

int main()
{
    Solution A;
    vector<vector<int>> rooms = {{1},{2},{3},{}};
    cout<<A.canVisitAllRooms(rooms);
}
*/


/*
//9.5 T1129 颜色交替的最短路径

struct node{ // 结构体
    int val;
    int layer;
    int flag;//1为红，2为蓝，0为都可以

    node(int val_ = 0,int layer_ = 0,int flag_ = 0):val(val_),layer(layer_),flag(flag_) {}
};

class Solution {
public:
    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& redEdges, vector<vector<int>>& blueEdges) {
        vector<vector<bool>> redMap(n,vector<bool>(n));  //把邻接表做出来
        vector<vector<bool>> blueMap(n,vector<bool>(n)); 
        vector<vector<int>> visited(n,vector<int>(2,INT_MAX));//是否访问过，并记录它的结果
        
        for(int i = 0;i < redEdges.size();i++)
        {
            redMap[redEdges[i][0]][redEdges[i][1]] = true;
        }
        for(int i = 0;i < blueEdges.size();i++)
        {
            blueMap[blueEdges[i][0]][blueEdges[i][1]] = true;        
        }

        queue<node> que;node top;
        que.push(node(0,0,0));visited[0][0] = 0;visited[0][1] = 0;

        while(!que.empty())
        {
            top = que.front();que.pop();
            switch(top.flag)
            {
                case 0://两个都可以（仅限于第一个）
                {
                    for(int i = 0;i < n;i++)
                    {
                        if(blueMap[top.val][i] && visited[i][0] == INT_MAX)
                        {
                            node tmp(i,top.layer + 1,2);
                            que.push(tmp);
                            visited[i][0] = top.layer + 1;//记录结果哟
                        }
                    }
                
                    for(int i = 0;i < n;i++)
                    {
                        if(redMap[top.val][i] && visited[i][1] == INT_MAX)
                        {
                            node tmp(i,top.layer + 1,1);
                            que.push(tmp);
                            visited[i][1] = top.layer + 1;//记录结果哟
                        }
                    }
                    break;
                }
                case 1://红，下一个是蓝
                {
                    for(int i = 0;i < n;i++)
                    {
                        if(blueMap[top.val][i] && visited[i][0] == INT_MAX)
                        {
                            node tmp(i,top.layer + 1,2);
                            que.push(tmp);
                            visited[i][0] = top.layer + 1;//记录结果哟
                        }
                    }
                    break;
                }
                case 2://蓝，下一个是红
                {
                    for(int i = 0;i < n;i++)
                    {
                        if(redMap[top.val][i] && visited[i][1] == INT_MAX)
                        {
                            node tmp(i,top.layer + 1,1);
                            que.push(tmp);
                            visited[i][1] = top.layer + 1;//记录结果哟
                        }
                    }
                    break;
                }
            }
        }

        vector<int> res(n);
        for(int i = 1;i < n;i++)
        {
            res[i] = min(visited[i][0],visited[i][1]);
            if(res[i] == INT_MAX) res[i] = -1;
        }
        return res;
    }
};
*/


/*
//9.5 T1376 通知所有员工所需要的时间

class Solution1 {//BFS
public:
    int numOfMinutes(int n, int headID, vector<int>& manager, vector<int>& informTime) {
        int i = 0,top = 0;
        vector<int> res(n);
        vector<vector<int>> map(n);
        for(i = 0;i < n;i++)//用邻接矩阵来储存
        {
            if(manager[i] == -1) top = i;
            else
            {
                map[manager[i]].push_back(i);
            }
        }
        queue<int> que;
        que.push(top);
        while(!que.empty())
        {
            top = que.front();que.pop();
            for(int j : map[top])
            {
                que.push(j);
                res[j] = res[top] + informTime[top];
            }
        }
        for(i = 1;i < n;i++)
        {
            if(res[i] > res[0]) res[0] = res[i];
        }
        return res[0];
    }
};

class Solution2 {//递归
public:
    vector<vector<int>> map;
    int numOfMinutes(int n, int headID, vector<int>& manager, vector<int>& informTime) {
        int i = 0,top = 0;
        vector<int> res(n);map.resize(n);
        for(i = 0;i < n;i++)//用邻接矩阵来储存
        {
            if(manager[i] == -1) top = i;
            else
            {
                map[manager[i]].push_back(i);
            }
        }

        return recur(top,informTime);//开始递归
    }

    int recur(int headID,vector<int>& informTime)//计算以headID为root得到的最少时间
    {
        if(map[headID].empty()) return 0;//如果是最底层员工

        int maxi = 0;
        for(int i : map[headID])
        {
            maxi = max(maxi,recur(i,informTime));//取子节点里面耗费时间最多的
        }
        return maxi + informTime[headID];
    }
};
*/



/*
// 9.6 T1446 重新规划路线

class Solution {
public:
    int minReorder(int n, vector<vector<int>>& connections) {
        vector<vector<int>> map(n);//用来储存点与点之间的关系
        vector<vector<bool>> map_(n);//用来判断来关系的方向是否错误
        vector<bool> visited(n);
        int i = 0,j = 0,res = 0;

        for(i = 0;i < n - 1;i++)//获得一个反向图
        {
            int a = connections[i][1],b = connections[i][0];
            map[a].push_back(b);map_[a].push_back(true);
            map[b].push_back(a);map_[b].push_back(false);
        }

        queue<int> que;int top = 0;
        que.push(0);visited[0] = true;

        while(!que.empty())
        {
            top = que.front();que.pop();
            for(i = 0;i < map[top].size();i++)
            {
                int tmp = map[top][i];
                if(!map_[top][i] && !visited[tmp])
                {
                    que.push(tmp);res++;
                    visited[tmp] = true;
                }
                else if(map[top][i] && !visited[tmp])
                {
                    que.push(tmp);
                    visited[tmp] = true;
                }
            }
        }

        return res;
    }
};
int main()
{
    Solution A;
    vector<vector<int>> c = {{1,0},{2,0}};
    cout<<A.minReorder(3,c);
}
*/


/*
//9.7 T797 所有可能的路径

class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        recur(0,graph);

        return res;
    }

    void recur(int root,vector<vector<int>>& graph)//int是当前头节点
    {
        path.push_back(root);
        for(int i : graph[root])
        {
            recur(i,graph);
        }
        if(root == graph.size() - 1) res.push_back(path);
        path.pop_back();//回溯
    }
};
*/


/*
//9.7 T1926 迷宫中离入口最近的出口

class Solution {//为防止重复遍历，把已经经过的节点设置为墙！
public:
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int row = maze.size(),col = maze[0].size(),res = 0;//行数，列数
        vector<vector<bool>> visited(row,vector<bool>(col));//是否已经访问过

        queue<vector<int>> que;que.push(entrance);visited[entrance[0]][entrance[1]] = true;
        vector<int> top(2);

        while(!que.empty())
        {
            int i = que.size();
            while(i--)
            {
                top = que.front();que.pop();
                int r = top[0],c = top[1];
                if((r != entrance[0] || c != entrance[1]) && (r == row - 1 || r == 0 || c == col - 1 || c == 0)) return res;

                if(r - 1 >= 0 && maze[r - 1][c] == '.' && !visited[r - 1][c])
                {
                    que.push({r - 1,c});
                    visited[r - 1][c] = true;
                }
                if(c + 1 < col && maze[r][c + 1] == '.' && !visited[r][c + 1])
                {
                    que.push({r,c + 1});
                    visited[r][c + 1] = true;
                }
                if(r + 1 < row && maze[r + 1][c] == '.' && !visited[r + 1][c])
                {
                    que.push({r + 1,c});
                    visited[r + 1][c] = true;
                }
                if(c - 1 >= 0 && maze[r][c - 1] == '.' && !visited[r][c - 1])
                {
                    que.push({r,c - 1});
                    visited[r][c - 1] = true;
                }
            }
            res++;
        }
        return -1;
    }
};
*/


/*
//9.9 T934 最短的桥

class Solution {
public:
    int shortestBridge(vector<vector<int>>& grid) {
        int n = grid.size(),i = 0,j = 0;
        pair<int,int> top;
        queue<pair<int,int>> que1;
        queue<pair<int,int>> que2;
        vector<int> dir1 = {1,0,-1,0};
        vector<int> dir2 = {0,1,0,-1};

        bool flag = true;
        for(i = 0;i < n && flag;i++)
        {
            for(j = 0;j < n && flag;j++)
            {
                if(grid[i][j] == 1)
                {
                    que1.push(pair<int,int>(i,j));
                    grid[i][j] = -1;//已经访问过
                    flag = false;
                }
            }
        }

        while(!que1.empty())
        {
            top = que1.front();que1.pop();
            que2.push(top);//放到第二个队列里面

            for(i = 0;i < 4;i++)
            {
                int a = top.first + dir1[i];
                int b = top.second + dir2[i];

                if(a >= 0 && a < n && b >= 0 && b < n && grid[a][b] == 1)
                {
                    que1.push(pair<int,int>(a,b));
                    grid[a][b] = -1;
                }
            }
        }

        int layer = -1;
        while(!que2.empty())
        {
            layer++;
            int len = que2.size();
            while(len--)
            {   
                top = que2.front();que2.pop();
                for(i = 0;i < 4;i++)
                {
                    int a = top.first + dir1[i];
                    int b = top.second + dir2[i];
                    if(a >= 0 && a < n && b >= 0 && b < n)
                    {
                        if(grid[a][b] == 0)
                        {
                            que2.push(pair<int,int>(a,b));
                            grid[a][b] = -1;
                        }
                        else if(grid[a][b] == 1)
                        {
                            return layer;
                        }
                    }
                }
            }
        }
        return 0;
    }
};
*/


/*
//9.10 T1192 查找集群内的关键连接

class Solution {
public:
    int num = 0;
    vector<vector<int>> map;
    vector<vector<int>> res;
    vector<int> dfn,low;//访问顺序 & 时间戳

    void tarjan(int u,int fa)
    {
        dfn[u] = low[u] = ++num;//初始化

        for(int v : map[u])
        {
            if(v == fa) continue;
            if(!dfn[v])
            {
                tarjan(v,u);//递归调用
                low[u] = min(low[u],low[v]);
                if(dfn[u] < low[v]) {res.push_back({u,v});}
            }
            else low[u] = min(low[u],dfn[v]);
        }
    }

    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
        map.resize(n);
        dfn.resize(n);
        low.resize(n);
        for(vector<int> connection : connections)//用邻接表来储存
        {
            int a = connection[0],b = connection[1];
            map[a].push_back(b);
            map[b].push_back(a);
        }

        for(int i = 0;i < n;i++)
        {
            if(!dfn[i]) tarjan(i,-1);
        }

        return res;
    }
};
*/


/*
//9.10 T433 最小基因变化

class Solution {
public:
    bool canVar(string start,string end)
    {
        int var = 0;
        for(int i = 0;i < 8;i++)
        {
            if(start[i] != end[i]) var++;//计算两者的区别基因个数
            if(var >= 2) return false;
        }
        return (var == 1);
    }

    int minMutation(string startGene, string endGene, vector<string>& bank) {
        int len = bank.size(),i = 0,j = 0;
        vector<vector<int>> adj(len);
        vector<bool> visited(len);
        queue<int> que;
        int top = 0;

        for(i = 0;i < len;i++) if(canVar(startGene,bank[i])) {que.push(i);visited[i] = true;}
        
        for(i = 0;i < len;i++)//初始化邻接表，把可以互相变化的基因存入邻接表中
        {
            for(j = i + 1;j < len;j++)
            {
                if(canVar(bank[i],bank[j]))
                {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }

        int layer = 0;
        while(!que.empty())//经典带距离的BFS过程
        {
            len = que.size();
            layer++;

            for(i = 0;i < len;i++)
            {
                top = que.front();que.pop();
                if(bank[top] == endGene) return layer;//找到末尾基因
                for(int j : adj[top])
                {
                    if(!visited[j])
                    {
                        visited[j] = true;
                        que.push(j);
                    }
                }
            }
        }
        return -1;
    }
};
*/


/*
//9.10 T127 单词接龙

class Solution {//与上一题代码十分相似，但是时间复杂度太高啦
public:
    bool canVar(string start,string end)
    {
        int var = 0;
        for(int i = 0;i < start.size();i++)
        {
            if(start[i] != end[i]) var++;//计算两者的区别基因个数
            if(var >= 2) return false;
        }
        return (var == 1);
    }

    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        if(find(wordList.begin(),wordList.end(),endWord) == wordList.end()) return false;

        int len = wordList.size(),i = 0,j = 0;
        vector<vector<int>> adj(len);
        vector<bool> visited(len);
        queue<int> que;
        int top = 0;

        for(i = 0;i < len;i++) if(canVar(beginWord,wordList[i])) {que.push(i);visited[i] = true;}
        
        for(i = 0;i < len;i++)//初始化邻接表，把可以互相变化的基因存入邻接表中
        {
            for(j = i + 1;j < len;j++)
            {
                if(canVar(wordList[i],wordList[j]))
                {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }

        int layer = 1;
        while(!que.empty())//经典带距离的BFS过程
        {
            len = que.size();
            layer++;

            for(i = 0;i < len;i++)
            {
                top = que.front();que.pop();
                if(wordList[top] == endWord) return layer;//找到末尾基因
                for(int j : adj[top])
                {
                    if(!visited[j])
                    {
                        visited[j] = true;
                        que.push(j);
                    }
                }
            }
        }
        return 0;
    }
};
*/


/*
//9.10 T1306 跳跃游戏3

class Solution {
public:
    bool canReach(vector<int>& arr, int start) {
        int top = 0,len = arr.size();
        queue<int> que;
        vector<bool> visited(len);
        que.push(start);

        while(!que.empty())
        {
            top = que.front();que.pop();
            if(arr[top] == 0) return true;
            int a = top - arr[top],b = top + arr[top];
            if(a >= 0 && a < len && !visited[a])
            {
                que.push(a);
                visited[a] = true;
            }
            if(b >= 0 && b < len && !visited[b])
            {
                que.push(b);
                visited[b] = true;
            }
        }
        return false;
    }
};
*/


/*
//9.11 T542 01矩阵

class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        pair<int,int> top;
        queue<pair<int,int>> que;
        int i,j,row = mat.size(),col = mat[0].size();
        vector<vector<bool>> visited(row,vector<bool>(col));//访问标志
        vector<int> dir = {1,0,-1,0,1};

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(mat[i][j] == 0)
                {
                    que.emplace((i,j)); //把是0的都放进去一起搜索
                    visited[i][j] = true; //已经访问过
                }
            }
        }

        int layer = -1;
        while(!que.empty())
        {
            int len = que.size();
            layer++;
            for(i = 0;i < len;i++)
            {
                top = que.front();que.pop();
                if(mat[top.first][top.second] == 1) mat[top.first][top.second] = layer; //返回层数

                for(j = 0;j < 4;j++)
                {
                    int a = top.first + dir[j],b = top.second + dir[j + 1];
                    if(a >= 0 && a < row && b >= 0 && b < col && !visited[a][b])
                    {
                        que.emplace(a,b);
                        visited[a][b] = true;
                    }
                }
            }
        }

        return mat;
    }
};
*/


/*
//9.12 T1091 二进制矩阵中的最短路径

class Solution {
public:
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        if(grid[0][0] == 1) return -1;

        int n = grid.size();
        pair<int,int> top;
        queue<pair<int,int>> que;
        vector<vector<bool>> visited(n,vector<bool>(n));
        vector<int> dir1 = {1,1,0,-1,-1,-1,0,1};//八个方向
        vector<int> dir2 = {0,1,1,1,0,-1,-1,-1};

        que.emplace(0,0);
        int layer = 0;
        while(!que.empty())
        {
            int len = que.size();
            layer++;
            while(len--)
            {
                top = que.front();que.pop();
                if(top.first == n - 1 && top.second == n - 1) return layer;
                for(int i = 0;i < 8;i++)
                {
                    int a = top.first + dir1[i],b = top.second + dir2[i];
                    if(a >= 0 && a < n && b >= 0 && b < n && !visited[a][b] && grid[a][b] == 0)
                    {
                        que.emplace(a,b);
                        visited[a][b] = true;
                    }
                }
            }
        }

        return -1;
    }
};
*/


/*
//9.13

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public: 
    int num = 0;//记录节点数量
    vector<vector<int>> adj;
    vector<TreeNode*> map;

    void order(TreeNode* root)//给每个节点编号，可以节省空间
    {
        if(!root) return;
        map.push_back(root);num++;
        order(root->left);
        order(root->right);
    }

    void buildMap(TreeNode* root)
    {
        if(!root) return;

        if(root->left)
        {
            adj[root->val].push_back(root->left->val);
            adj[root->left->val].push_back(root->val);
        }
        if(root->right)
        {
            adj[root->val].push_back(root->right->val);
            adj[root->right->val].push_back(root->val);
        }

        buildMap(root->left);
        buildMap(root->right);
    }

    vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
        vector<int> res;
        adj.resize(501);
        vector<bool> visited(501);

        buildMap(root);//建图

        queue<int> que;int top;
        que.push(target->val);visited[target->val] = true;
        int layer = -1;
        
        while(!que.empty())
        {
            int len = que.size();
            layer++;
            if(layer == k) break;
            
            for(int i = 0;i < len;i++)
            {
                top = que.front();que.pop();
                for(int j : adj[top])
                {
                    if(!visited[j])
                    {
                        que.push(j);
                        visited[j] = true;
                    }
                }
            }
        }

        while(!que.empty())
        {
            top = que.front();que.pop();
            res.push_back(top);
        }

        return res;
    }
};
*/


/*
//9.13 T200 岛屿数量

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int row = grid.size(),col = grid[0].size();
        queue<pair<int,int>> que;pair<int,int> top;
        vector<int> dir = {1,0,-1,0,1};
        int res = 0;

        for(int i = 0;i < row;i++)
        {
            for(int j = 0;j < col;j++)
            {
                if(grid[i][j] == '1')
                {
                    res++;
                    que.emplace(i,j);grid[i][j] = '0';
                    while(!que.empty())
                    {
                        top = que.front();que.pop();
                        for(int k = 0;k < 4;k++)
                        {
                            int a = top.first + dir[k];
                            int b = top.second + dir[k + 1];

                            if(a >= 0 && a < row && b >= 0 && b < col && grid[a][b] == '1')
                            {
                                que.emplace(a,b);
                                grid[a][b] = '0';
                            }
                        }
                    }
                }
            }
        }
        return res;
    }
};
*/


/*
//9.15 T1020 飞地的数量

class Solution1 {//BFS
public:
    int numEnclaves(vector<vector<int>>& grid) {
        vector<int> dir = {1,0,-1,0,1};
        int row = grid.size(),col = grid[0].size(),i,j,res = 0;
        queue<pair<int,int>> que;pair<int,int> top;

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1)
                {
                    top.first = i;top.second = j;
                    que.emplace(i,j);
                    grid[i][j] = 0; //已访问过

                    int num = 1; //计算这一块的数量
                    bool flag = false; //是否是飞地
                    while(!que.empty())
                    {
                        top = que.front();que.pop();
                        int a = top.first,b = top.second;
                        if(a == 0 || a == row - 1 || b == 0 || b == col - 1)
                        {
                            flag = true;
                        }
                        for(int k = 0;k < 4;k++)
                        {
                            int c = a + dir[k];
                            int d = b + dir[k + 1];
                            if(c >= 0 && c < row && d >= 0 && d < col && grid[c][d] == 1)
                            {
                                que.emplace(c,d);
                                grid[c][d] = 0; //已访问过
                                num++; //飞地数量加1
                            }
                        }
                    }

                    if(!flag) res += num;
                }
            }
        }

        return res;
    }
};



class Solution {//并查集
public:
    vector<int> fa,rank;
    vector<bool> onEdge;

    int Find(int idx) //找到父节点
    {
        if(fa[idx] == idx) return idx;
        
        fa[idx] = Find(fa[idx]);
        return fa[idx];
    }

    void Union(int idx1,int idx2) //按秩合并
    {
        int f1 = Find(idx1),f2 = Find(idx2);

        if(rank[f1] <= rank[f2])
        {
            fa[f1] = f2; //合并
            onEdge[f2] = onEdge[f2] || onEdge[f1]; //是否在边界上
        }

        else
        {
            fa[f2] = f1;
            onEdge[f1] = onEdge[f2] || onEdge[f1];
        }

        if(rank[f1] == rank[f2] && f1 != f2) //秩相同，rank加1
        {
            rank[f2]++;
        }
    }


    int numEnclaves(vector<vector<int>>& grid) {
        int row = grid.size(),col = grid[0].size(),i,j;
        fa.resize(row * col); //父节点数组
        rank.resize(row * col); //每个集的秩
        onEdge.resize(row * col,false); //每个点是否在边上

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1)
                {
                    int idx = col * i + j; //手动哈希表
                    fa[idx] = idx;
                    if(i == 0 || i == row - 1 || j == 0 || j == col - 1)
                    {
                        onEdge[idx] = true;
                    }
                }
            }
        }

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1)
                {
                    int idx = col * i + j;
                    //合并两个点
                    if(i + 1 < row && grid[i + 1][j] == 1) Union(idx,idx + col); 
                    if(j + 1 < col && grid[i][j + 1] == 1) Union(idx,idx + 1);
                    if(i - 1 >= 0 && grid[i - 1][j] == 1) Union(idx,idx - col);
                    if(j - 1 >= 0 && grid[i][j - 1] == 1) Union(idx,idx - 1);
                }
            }
        }
        //计算结果
        int res = 0;
        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1 && !onEdge[Find(col*i + j)]) res++;
            }
        }
        return res;
    }
};
*/


/*
//9.16 T695 岛屿的最大面积

class Solution {
public:
    vector<int> fa; //每个点的父节点
    vector<int> area; //每个点的区域面积
    //vector<int> rank; //rank数组

    int Find(int idx) //找到领导节点
    {
        if(fa[idx] == idx) return idx;

        fa[idx] = Find(fa[idx]);
        return fa[idx];
    }

    void Union(int idx1,int idx2) //合并两个点
    {
        int f1 = Find(idx1),f2 = Find(idx2);
        if(f1 == f2) return;

        fa[f1] = f2;
        area[f2] += area[f1]; //面积相加
    }

    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int row = grid.size(),col = grid[0].size(),i,j;
        fa.resize(row * col);area.resize(row * col);//初始化参数

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1)
                {
                    int idx = col * i + j;
                    fa[idx] = idx; //领导节点是自己
                    area[idx] = 1; //面积先定为1
                }
            }
        }

        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(grid[i][j] == 1)
                {
                    int idx = col * i + j;
                    if(i + 1 < row && grid[i + 1][j] == 1) Union(idx,idx + col); //合并下边的
                    if(j + 1 < col && grid[i][j + 1] == 1) Union(idx,idx + 1);   //合并右边的
                }
            }
        }

        return *max_element(area.begin(),area.end());
    }
};
*/


/*
//9.16 T417 太平洋大西洋水流问题

class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        int row = heights.size(),col = heights[0].size(),i,j;
        vector<vector<int>> Atl(row,vector<int>(col)); //能否流到右下,=0表示没访问过，=1表示可以，=-1表示不行
        vector<vector<int>> Pac = Atl; //能否游到左上
        vector<int> dir = {1,0,-1,0,1};

        queue<pair<int,int>> que;pair<int,int> top;
        //先看看大西洋
        for(i = 0;i < row;i++)//超级源点
        {
            que.emplace(i,col - 1);
            Atl[i][col - 1] = 1;
        }
        for(j = 0;j < col - 1;j++)
        {
            que.emplace(row - 1,j);
            Atl[row - 1][j] = 1;
        }

        while(!que.empty())
        {
            top = que.front();que.pop();
            int height = heights[top.first][top.second]; //当前节点高度
            for(int k = 0;k < 4;k++)
            {
                int a = top.first + dir[k],b = top.second + dir[k + 1];
                if(a >= 0 && a < row && b >= 0 && b < col )
                {
                    if(Atl[a][b] == 0 && heights[a][b] >= height)
                    {
                        que.emplace(a,b);
                        Atl[a][b] = 1;
                    }
                }
            }
        }
        que = queue<pair<int,int>>(); //清空队列

        //太平洋
        for(j = 0;j < col;j++)
        {
            que.emplace(0,j);
            Pac[0][j] = 1;
        }
        for(i = 1;i < row;i++)
        {
            que.emplace(i,0);
            Pac[i][0] = 1;
        }

        while(!que.empty())
        {
            top = que.front();que.pop();
            int height = heights[top.first][top.second]; //当前节点高度
            for(int k = 0;k < 4;k++)
            {
                int a = top.first + dir[k],b = top.second + dir[k + 1];
                if(a >= 0 && a < row && b >= 0 && b < col)
                {
                    if(Pac[a][b] == 0 && heights[a][b] >= height)
                    {
                        que.emplace(a,b);
                        Pac[a][b] = 1;
                    }
                }
            }
        }
        
        vector<vector<int>> res;
        for(i = 0;i < row;i++)
        {
            for(j = 0;j < col;j++)
            {
                if(Pac[i][j] == 1 && Atl[i][j] == 1)
                {
                    res.push_back({i,j});
                }
            }
        }

        return res;
    }
};
*/


/*
//9.17 T997 小镇法官

class Solution {
public:
    int findJudge(int n, vector<vector<int>>& trust) {
        vector<int> inEdge(n),outEdge(n);
        
        for(auto relate : trust)
        {
            outEdge[relate[0] - 1]++;
            inEdge[relate[1] - 1]++;
        }

        for(int i = 0;i < n;i++)
        {
            if(outEdge[i] == 0 && inEdge[i] == n - 1) return i + 1;
        }
        return -1;
    }
};
*/


/*
//9.18 T1557 

class Solution {
public:
    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
        vector<int> inEdge(n);
        vector<int> res;
        for(auto edge : edges) inEdge[edge[1]]++;
        for(int i = 0;i < n;i++)
        {
            if(inEdge[i] == 0) res.push_back(i);
        }

        return res;
    }
};
*/


/*
//9.18 T1615 网络最大秩 ?需要检查

class Solution {
public:
    int maximalNetworkRank(int n, vector<vector<int>>& roads) {
        vector<vector<bool>> mat(n,vector<bool>(n,false)); //邻接表储存之间是否有边
        vector<int> edge(n);

        for(auto road : roads)
        {
            int a = road[0],b = road[1];
            mat[a][b] = true;mat[b][a] = true;
            edge[a]++;edge[b]++;
        } //存邻接表，度数表

        int maxi = (edge[0] > edge[1]) ? 0 : 1,maxi_ = maxi ^ 0 ^ 1;

        vector<int> first,second; //储存最大的点，第二大的点
        for(int i = 2;i < n;i++) 
        {
            if(edge[i] > edge[maxi]) maxi = i; //最大值序号更新
            else if(edge[i] > edge[maxi_]) maxi_ = i; //次大值序号更新
        }

        for(int i = 0;i < n;i++)
        {
            if(edge[i] == edge[maxi]) first.push_back(i);
            else if(edge[i] == edge[maxi_]) second.push_back(i);
        }

        if(first.size() == 1) //遍历second，找到最大的
        {
            for(int j : second)
            {
                if(!mat[maxi][j]) return edge[maxi] + edge[maxi_]; //有没边的
            }
            return edge[maxi] + edge[maxi_] - 1; //都有边
        }

        else
        {
            for(int i : first)
            {
                for(int j : first)
                {
                    if(i != j && !mat[i][j]) return 2*edge[maxi];
                }
            }

            return 2*edge[maxi] - 1;
        }
    }
};
*/


/*
//9.19 T785 判断二分图

class Solution1 { //BFS写法
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int len = graph.size(),i; //表的长度，也就是点的个数
        vector<int> log(len,-1); //记录谁在哪个集合中
        queue<pair<int,int>> que; //队列，用来循环
        pair<int,int> top;

        for(i = 0;i < len;i++) //初始化
        {
            if(!graph[i].empty())
            {
                que.emplace(i,1);
                log[i] = 1;
                break;
            }
        }

        for(i = 0;i < len;i++)
        {   
            while(!que.empty())
            {
                top = que.front();que.pop();
                int c = 3 - top.second; //应该是集合几

                for(int i : graph[top.first])
                {
                    if(log[i] == -1) //未访问过
                    {
                        log[i] = c; //记录
                        que.emplace(i,c); //进入队列
                    } 
                    else //已经访问过
                    {
                        if(log[i] != c) return false;
                    }
                }
            }
            for(i = 0;i < len;i++) //访问下一个分支
            {
                if(log[i] == -1 && !graph[i].empty())
                {
                    que.emplace(i,1);
                    log[i] = 1;
                    break;
                }
            }
        }
        return true;
    }
};
*/


/*
//9.23 T839

class Solution { //暴力并查集
public:
    vector<int> fa; //储存父亲节点

    int Find(int idx)
    {
        if(fa[idx] == idx) return idx;

        fa[idx] = Find(fa[idx]);
        return fa[idx];
    }

    void Union(int idx1,int idx2) //合并
    {
        fa[Find(idx1)] = Find(idx2);
    }

    bool isSimilar(string str1,string str2) //判断函数
    {
        if(str1 == str2) return true; //相等，直接返回

        int dif = 0; //str1、str2有多少区别
        for(int i = 0;i < str1.size();i++)
        {
            if(str1[i] != str2[i]) dif++;
        }

        return (dif == 2); //dif不是2，直接返回错误
    }

    int numSimilarGroups(vector<string>& strs) {
        int len = strs.size(),i,j,res = 0; //数组长度
        fa.resize(len);
        for(i = 0;i < len;i++) fa[i] = i; //初始化父亲数组

        for(i = 0;i < len;i++)
        {
            for(j = i + 1;j < len;j++)
            {
                if(isSimilar(strs[i],strs[j])) //相似，就放在一起
                {
                    Union(i,j);
                }
            }
        }

        for(i = 0;i < len;i++)
        {
            if(fa[i] == i) res++;
        }
        return res;
    }
};
*/


//10.30 T1514

class Solution {
public:
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start_node, int end_node) {
        vector<vector<double>> mat(n, vector<double>(n, 0)); //邻接矩阵储存，没有边则是0
        vector<bool> isFind(n, false); //是否有答案数组，为true则是已经找到最短路
        vector<double> res(n); //表示已经找到答案的大小
        int len = succProb.size(); //边的个数
        for(int i = 0;i < len;i++)
        {
            int a = edges[i][0], b = edges[i][1];
            mat[a][b] = succProb[i];mat[b][a] = succProb[i];
        }

        for(int i = 0;i < n;i++) mat[i][i] = 1;
        for(int i = 0;i < n;i++)
        {
            res[i] = mat[i][start_node];
            mat[i][i] = 1;
        } //初始化答案数组
        
        //开始算法
        isFind[start_node] = true;
        for(int i = 0;i < n;i++)
        {
            double maxi = -100, max_idx = 0;
            for(int j = 0;j < n;j++)
            {
                if(!isFind[j] && res[j] > maxi) //没被找过且在这里面概率最大
                {
                    maxi = res[j];
                    max_idx = j;
                }
            }

            isFind[max_idx] = true;
            for(int j = 0;j < n;j++)
            {             
                double tmp = res[max_idx] * mat[max_idx][j];
                if(tmp > res[j]) res[j] = tmp; //更新              
            }
        }
        return res[end_node];
    }
};
int main()
{
    Solution A;
    vector<vector<int>> edges = {{0,1},{1,2},{0,2}};
    vector<double> succProb = {0.5, 0.5, 0.3};
    cout << A.maxProbability(3, edges, succProb, 0, 2);
}