#include <bits/stdc++.h>

using namespace std;


/*
//8.24 T1267 统计参与通信的服务器
// 这里有一幅服务器分布图，服务器的位置标识在 m * n 的整数矩阵网格 grid 中，1 表示单元格上有服务器，0 表示没有。
// 如果两台服务器位于同一行或者同一列，我们就认为它们之间可以进行通信。请你统计并返回能够与至少一台其他服务器进行通信的服务器的数量。

class Solution {
public:
    int countServers(vector<vector<int>>& grid) {
        unordered_map<int,int> row,col;
        int r = grid.size(),c = grid[0].size();
        int i = 0, j = 0,res = 0;

        for(i = 0;i < r;i++)
        {
            for(j = 0;j < c;j++)
            {
                if(grid[i][j] == 1) {row[i]++;col[j]++;}
            }
        }

        for(i = 0;i < r;i++)
        {
            for(j = 0;j < c;j++)
            {
                if(grid[i][j] == 1 && (row[i] > 1 || col[j] > 1)) res++;
            }
        }

        return res;
    }
};

int main()
{
    Solution A;
    vector<vector<int>> grid = {{0,1},{1,1}};
    cout<<A.countServers(grid);
}
*/


/*
//8.24 T849 到最近的人的最大距离
// 给你一个数组 seats 表示一排座位，其中 seats[i] = 1 代表有人坐在第 i 个座位上，seats[i] = 0 代表座位 i 上是空的（下标从 0 开始）。
// 至少有一个空座位，且至少有一人已经坐在座位上。
// 亚历克斯希望坐在一个能够使他与离他最近的人之间的距离达到最大化的座位上。返回他到离他最近的人的最大距离。

class Solution {
public:
    int maxDistToClosest(vector<int>& seats) {
        int i = 0,len = seats.size(),left = -1,right = 0;
        int res = 0;

        for(i = 0;i < len;i++)
        {
            if(seats[i] == 1)
            {
                right = i;

                if(left == -1) res = right;
                else res = max(res,(right - left)/2);
                left = i;
            }
        }

        return max(res,len - 1 - right);
    }
};

int main()
{
    Solution A;
    vector<int> nums = {1,0,0,0,1,0,1};
    cout<<A.maxDistToClosest(nums);
}
*/


/*
//8.25 T1448 统计二叉树中好节点的个数
// 给你一棵根为 root 的二叉树，请你返回二叉树中好节点的数目。
// 「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int res = 0;
    int goodNodes(TreeNode* root) {
        if(!root) return 0;

        recur(root,root->val);
        return res;
    }

    void recur(TreeNode* root,int maxi)
    {
        if(!root) return;

        if(root->val >= maxi)
        {
            maxi = root->val;res++;
        }

        recur(root->left,maxi);
        recur(root->right,maxi);
    }
};
*/


/*
//8.26 T228 汇总区间
// 给定一个  无重复元素 的 有序 整数数组 nums 。
// 返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表 。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，
// 并且不存在属于某个范围但不属于 nums 的数字 x 。列表中的每个区间范围 [a,b] 应该按如下格式输出：
// "a->b" ，如果 a != b
// "a" ，如果 a == b

class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        int l = 0,r = 0,len = nums.size();
        vector<string> res;

        for(r = 0;r < nums.size();r++)
        {
            //如果nums[r] + 1 == nums[r + 1]，r就++
            if((r == nums.size() - 1) || (nums[r] + 1 < nums[r + 1]))  //如果不是最后或者不是等差数列，就进行处理
            {
                if(l == r) {res.push_back(to_string(nums[l]));l++;}

                else
                {
                    string s;
                    s += to_string(nums[l]);s += "->";s += to_string(nums[r]);
                    l = r + 1;
                    res.push_back(s);
                }
            }
        }

        return res;
    }
};

int main()
{
    Solution A;
    vector<int> nums = {-1,0,1,2,3,4,6,8,9};
    vector<string> res = A.summaryRanges(nums); 
    for(string s:res) cout << s <<' ';
}
*/


/*
//8.27 T56 合并区间
//以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
// 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
// 输出：[[1,6],[8,10],[15,18]]
// 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

bool cmp(vector<int>& a,vector<int>& b)
{
    return a[0] < b[0];
}

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),cmp);
        int j = 0,len = intervals.size();

        vector<vector<int>> res;
        res.push_back(intervals[0]);

        for(j = 1;j < len;j++)
        {
            if(res.back()[1] < intervals[j][0])//说明该区间一定不能与前面区间合并
            {
                res.push_back(intervals[j]);
            }
            else
            {
                res.back()[1] = max(res.back()[1],intervals[j][1]);//更新区间的最大值（也就是合并区间）
            }
        }

        return res;
    }
};

int main()
{
    Solution A;
    vector<vector<int>> inter = {{1,3},{2,6},{8,10},{15,18}};
    vector<vector<int>> res = A.merge(inter);
}
*/


/*
//8.28 T57 插入区间
//给你一个 无重叠的 ，按照区间起始端点排序的区间列表。在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

class Solution {
public:
    int bin(vector<vector<int>>& intervals,vector<int>& newInterval)//二分法，返回第一个比目标大的位置
    {
        int len = intervals.size();int left = 0,right = len - 1;
        if(intervals.back()[0] < newInterval[0]) return len;

        while(left < right)
        {
            int mid = (left + right)/2;
            if(intervals[mid][0] < newInterval[0]) {left = mid + 1;}
            else if(intervals[mid][0] > newInterval[0]) {right = mid;}
            else return mid;
        }
        return right;
    }

    void insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        if(intervals.empty()) return {newInterval};
        int right = bin(intervals,newInterval) - 1;

        //先处理前一个的影响
        int i = 0;
        if(right >= 0)
        {
            if(intervals[right][1] >= newInterval[1]) return;//return intervals;//不用插入的情况
            else if(intervals[right][1] >= newInterval[0]) {intervals[right][1] = newInterval[1];i = right + 1;}
            else {intervals.insert(intervals.begin() + right + 1,newInterval);i = right + 2;}
        }
        else if(right < 0) {intervals.insert(intervals.begin() + right + 1,newInterval);i = right + 2;}

        int maxi = 0;
        for(;i < intervals.size() && newInterval[1] >= intervals[i][0];)//这里面的都是可以合并的
        {
            intervals[i - 1][1] = max(intervals[i - 1][1],intervals[i][1]);
            intervals.erase(intervals.begin() + i);
        }
    }
};

int main()
{
    Solution A;
    vector<vector<int>> a = {{1,3},{6,9}};
    vector<int> b = {2,5};
    A.insert(a,b);
}
*/


/*
//8.29 T823 带因子的二叉树
// 给出一个含有不重复整数元素的数组 arr ，每个整数 arr[i] 均大于 1。
// 用这些整数来构建二叉树，每个整数可以使用任意次数。其中：每个非叶结点的值应等于它的两个子结点的值的乘积。
// 满足条件的二叉树一共有多少个？答案可能很大，返回 对 109 + 7 取余 的结。

const int MAX = 1e9 + 7;
class Solution {
public:
    int numFactoredBinaryTrees(vector<int>& arr) {
        int k = 0,len = arr.size();
        vector<int> res(len,1);//res[k]表示以第k个结点为根的子树个数
        sort(arr.begin(),arr.end());//排序
        long long a = 1;//最终结果

        for(k = 1;k < len;k++)
        {
            int i = 0,j = k - 1;
            while(i <= j)
            {   
                if(arr[i] < double(arr[k]) / arr[j]) i++;     //运用双指针
                else if(arr[i] > double(arr[k]) / arr[j]) j--;
                else //arr[i]*arr[j] = arr[k]
                {
                    res[k] += (1 + (i < j)) * res[i] * res[j];//递归，注意i==j和i<j的差别
                    i++;
                }
            }
            a += res[k];
        }
        return a % MAX;
    }
};
*/


/*
//8.30 T1654 到家最少跳跃数
// 有一只跳蚤的家在数轴上的位置 x 处。请你帮助它从位置 0 出发，到达它的家。
// 它可以 往前 跳恰好 a 个位置（即往右跳）。它可以 往后 跳恰好 b 个位置（即往左跳）。它不能 连续 往后跳 2 次。
// 它不能跳到任何 forbidden 数组中的位置。跳蚤可以往前跳 超过 它的家的位置，但是它 不能跳到负整数 的位置。
// 给你一个整数数组 forbidden ，其中 forbidden[i] 是跳蚤不能跳到的位置，
// 同时给你整数 a， b 和 x ，请你返回跳蚤到家的最少跳跃次数。如果没有恰好到达 x 的可行方案，请你返回 -1 。

const int MAX = 6005;
//一个节点的结构体
struct Node{
    int val;  //值
    int layer;//BFS层数
    bool flag;//上一步向左为true，向右为false

    Node(int val_ = 0,int layer_ = 0,bool flag_ = false):val(val_),layer(layer_),flag(flag_){}//初始化列表
};

class Solution {
public:
    int minimumJumps(vector<int>& forbidden, int a, int b, int x) {
        unordered_set<int> unset;
        for(int forbid:forbidden) {unset.emplace(forbid);}//增加已经访问的
        queue<Node> que;
        Node top,left,right;

        //初始状态，把起点放进队列que
        que.push(Node(0,0));unset.emplace(0);//节点已经访问过

        while(!que.empty())//队列不为空，继续搜索
        {
            top = que.front();que.pop();//取出第一个并从队列中移除
            if(top.val == x) return top.layer;

            left.val = top.val - b;right.val = top.val + a;
            if(right.val >= 0 && right.val <= MAX && unset.find(right.val) == unset.end())//符合标准且没访问过
            {
                right.layer = top.layer + 1;//更新层数
                que.push(right);     //加入队列
                unset.emplace(right.val);//已访问过 
            }
            if(left.val >= 0 && left.val <= MAX && !top.flag && unset.find(left.val) == unset.end())
            {
                left.layer = top.layer + 1;//更新层数
                left.flag = true;   //表示上一次是向左
                que.push(left);     //加入队列     
            }
        }

        return -1;
    }
};

int main()
{
    Solution A;
    vector<int> forbidden = {8,3,16,6,12,20};
    cout<<A.minimumJumps(forbidden,15,13,11);
}
*/



/*
//8.31 T1761 联通三元组的最小度数

class Solution {
public:
    int minTrioDegree(int n, vector<vector<int>>& edges) {
        int i,j,k,mini = 1e9;
        vector<vector<bool>> edgeMat(n,vector<bool>(n));
        vector<int> degree(n);
        for(i = 0;i < edges.size();i++)//邻接表，同时计算度数
        {
            edgeMat[edges[i][0] - 1][edges[i][1] - 1] = true;
            edgeMat[edges[i][1] - 1][edges[i][0] - 1] = true;
            degree[edges[i][0] - 1]++;degree[edges[i][1] - 1]++;
        }

        bool flag = false;//判断是否有三元数组
        for(i = 0;i < n;i++)
        {
            for(j = i + 1;j < n;j++)
            {
                for(k = j + 1;k < n;k++)
                {
                    if(edgeMat[i][j] && edgeMat[i][k] && edgeMat[k][j])
                    {
                        flag = true;
                        mini = min(mini,degree[i] + degree[j] + degree[k] - 6);
                    }
                }
            }
        }
        
        return flag ? mini : -1;
    }
};
*/


/*
class Solution {
public:
    int captureForts(vector<int>& forts) {
        int maxi = 0,i = 0;
        for(;i < forts.size();i++)
        {
            if(forts[i] == 0) continue;//数组最跳过前面的0

            int j = i + 1;        
            if(j >= forts.size()) break;//判断出界1

            for(;j < forts.size() && forts[j] == 0;j++) {}
            if(j >= forts.size()) break;//判断出界2
            if(forts[i] * forts[j] == -1) maxi = max(maxi,j - i - 1);//i、j记录1或-1的位置
            i = j - 1;//继续循环
        }

        return maxi;
    }
};

int main()
{
    Solution A;vector<int> forts = {1,-1,1,0};
    cout<<A.captureForts(forts);
}
*/


/*
//9.5 T2605 从两个数字数组里生成最小数字

class Solution {
public:
    int minNumber(vector<int>& nums1, vector<int>& nums2) {
        vector<bool> arr1(10),arr2(10);
        int mini1 = 10,mini2 = 10;

        for(int i : nums1)
        {
            mini1 = min(mini1,i);//找最小元素
            arr1[i] = true;      //找有哪些元素，为了判断是否有相同元素
        }
        for(int j : nums2)
        {
            mini2 = min(mini2,j);
            arr2[j] = true;
        }
        for(int i = 0;i < 10;i++) 
        {
            if(arr1[i] && arr2[i]) return i;//有相同元素，则返回第一个找到的（最小的）那个
        }

        return (mini1 < mini2) ? (10*mini1 + mini2) : (10*mini2 + mini1);//没有相同元素，返回ab或ba
    }
};
*/


/*
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int depth(TreeNode* root)
    {
        if(!root) return 0;
        return max(depth(root->left),depth(root->right)) + 1;
    }
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        if(!root) return root;

        int d1 = depth(root->left),d2 = depth(root->right);
        if(d1 == d2) return root;
        else if(d1 > d2) return lcaDeepestLeaves(root->left);
        else return lcaDeepestLeaves(root->right);
    }
};
*/


/*
//9.7 T2594 修车的最少时间

class Solution {
public:
    using ll = long long;
    long long repairCars(vector<int>& ranks, int cars) {
        ll res = 0;int n = ranks.size();
        vector<double> ranks1(n);
        for(int i = 0;i < n;i++) ranks1[i] = pow(ranks[i],-0.5);

        long long mini,maxi;
        double frac = 0;
        for(double i : ranks1) frac += i;
        ll l = floor(pow(double(cars)/frac,2));
        ll r = floor(pow(double(cars)/frac + n,2));
        auto check = [&](ll m) {
            ll cnt = 0;
            for (auto x : ranks) {
                cnt += sqrt(m / x);
            }
            return cnt >= cars;
        };
        while (l < r) {
            ll m = l + r >> 1;
            if (check(m)) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }
};
*/


/*
//9.9 T207 课程表

class Solution {//经典拓扑排序
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        if(prerequisites.empty()) return false;

        vector<vector<int>> map(numCourses);vector<int> deg(numCourses);
        stack<int> stk;int top = 0,res = 0;
        for(vector<int> tmp : prerequisites)//map[i] = j表示i是j的前驱
        {
            deg[tmp[0]]++;
            map[tmp[1]].push_back(tmp[0]);
        }
        for(int i = 0;i < numCourses;i++)
        {
            if(deg[i] == 0)
            {
                stk.push(i);//将入度为0的课程加入stack中
            }
        }

        while(!stk.empty())
        {
            top = stk.top();stk.pop();
            res++;//有一门课可以上

            for(int j : map[top])
            {
                if(--deg[j] == 0) stk.push(j);
            }
        }

        return (res == numCourses);
    }
};
*/


/*
//9.10 T210 课程表2
class Solution1 {//经典拓扑排序
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> res;
        int flag = 0;

        vector<vector<int>> map(numCourses);vector<int> deg(numCourses);
        stack<int> stk;int top = 0;
        for(vector<int> tmp : prerequisites)//map[i] = j表示i是j的前驱
        {
            deg[tmp[0]]++;
            map[tmp[1]].push_back(tmp[0]);
        }
        for(int i = 0;i < numCourses;i++)
        {
            if(deg[i] == 0)
            {
                stk.push(i);//将入度为0的课程加入stack中
            }
        }

        while(!stk.empty())
        {
            top = stk.top();stk.pop();
            res.push_back(top);//有一门课可以上
            flag++;

            for(int j : map[top])
            {
                if(--deg[j] == 0) stk.push(j);
            }
        }

        if(flag == numCourses) return res;
        else return {};
    }
};
*/


/*
class Solution2 { //用DFS+时间戳来写
public:
    vector<vector<int>> adj; //邻接表储存
    vector<int> start, finish; //开始和结束时间戳
    vector<int> edge; //每个点的入度，入度为0的可以作为头节点进循环
    vector<int> res; //答案数组
    int idx; //时间戳
    bool explore(int i, int &t) //输入是头节点
    {
        start[i] = t;
        t++;
        for(int next : adj[i])
        {
            if(start[next] != 0 && finish[next] == 0) return false;
            else if(start[next] == 0 && finish[next] == 0) //没有被访问过
            {
                bool flag = explore(next, t);
            }
        }
        finish[i] = t;
        res[idx--] = i;
        t++;
        return true;
    }

    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        adj.resize(numCourses); start.resize(numCourses); finish.resize(numCourses); edge.resize(numCourses); res.resize(numCourses); idx = numCourses - 1;

        for(auto pre : prerequisites)
        {
            int a = pre[0], b = pre[1];
            adj[b].push_back(a);
            edge[b]++;
        }
        //开始拓扑排序
        bool flag1 = false;
        int time = 1; //时间戳
        for(int i = 0;i < numCourses;i++)
        {
            if(edge[i] == 0)
            {
                flag1 = true;
                if(!explore(i, time)) return {}; //不能形成环
            }
        }
        if(!flag1 || idx != -1) return {}; //没有入度为0的点且遍历完全
        return res;
    }
};
*/



/*
//9.10 T630 课程表 3

class Solution {
public:
    static bool cmp(vector<int> a,vector<int> b)
    {
        return a[1] < b[1];
    }
    int scheduleCourse(vector<vector<int>>& courses) {
        sort(courses.begin(),courses.end(),cmp);//选择顺序确定
        int i = 0,j = 0;

        priority_queue<int,vector<int>,less<int>> que;
        int time = 0;

        for(auto course : courses)
        {
            que.push(course[0]);
            time += course[0];
            if(time > course[1])
            {
                while(time > course[1])
                {
                    time -= que.top();
                    que.pop();
                }
            }
        }

        return que.size();
    }
};
*/


/*
//9.12 T1462 课程表4

class Solution {
public:
    vector<vector<int>> adj;
    bool DFS(int u,int v,vector<vector<int>>& mat)
    {
        if(mat[u][v] != 0) return (mat[u][v] == 1);

        for(int son : adj[u])
        {
            if(DFS(son,v,mat))
            {
                mat[u][v] = 1;
                return true;
            }
        }

        mat[u][v] = -1;
        return false;
    }

    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        adj.resize(numCourses);
        vector<vector<int>> mat(numCourses,vector<bool>(numCourses));//是1表示已访问过并且是，-1表示已访问过但是不是，0表示没访问过

        for(auto p : prerequisites)//邻接矩阵存关系，邻接表存真正的表
        {
            mat[p[0]][p[1]] = 1;
            adj[p[0]].push_back(p[1]);
        }

        vector<bool> res;
        for(auto query : queries)
        {
            res.push_back(DFS(query[0],query[1],mat));
        }

        return res;
    }
};
*/


//9.13 T2596 
/*
class Solution {
public:
    bool checkValidGrid(vector<vector<int>>& grid) {
        int n = grid.size(),a = 0,b = 0;
        bool flag = true;
        if(grid[a][b] != 0) return false;

        vector<int> dir1 = {2,2,1,-1,-2,-2,-1,1};
        vector<int> dir2 = {-1,1,2,2,1,-1,-2,-2};
        int i = 0,c,d;
        
        while(i < n * n - 1)
        {
            bool flag = true;
            for(int j = 0;j < 8 && flag;j++)
            {
                c = a + dir1[j];
                d = b + dir2[j];
                if(c >= 0 && c < n && d >= 0 && d < n && grid[c][d] == i + 1) {flag = false;break;}
            }
            if(flag) return false;
            a = c;b = d;i++;
        }
        return true;
    }
};
*/


/*
//9.14 T1222 可以攻击国王的王后

class Solution {
public:
    vector<vector<int>> queensAttacktheKing(vector<vector<int>>& queens, vector<int>& king) {
        vector<int> dir1 = {1,1,0,-1,-1,-1,0,1};
        vector<int> dir2 = {0,1,1,1,0,-1,-1,-1};

        map<pair<int,int>,int> map1;
        for(auto queen : queens)
        {
            map1.insert(queen[0],queen[1]);
        }
        queens.clear();

        int x = king[0],y = king[1];
        for(int i = 0;i < 8;i++)
        {
            while(x >= 0 && x < 8 && y >= 0 && y < 8)
            {
                x += dir1[i];y += dir2[i];
                if(map1.count(x,y))
                {
                    queens.push_back({x,y});
                    break;
                }
            }
        }
        return queens;
    }
};
*/


/*
//9.17 T213 打家劫舍2

class Solution {
public:
    int rob1(vector<int>& nums) //打家劫舍1
    {
        int n = nums.size();
        vector<int> dp(n + 1);
        dp[1] = nums[0];
        for(int i = 2;i <= n;i++)
        {
            dp[i] = max(dp[i - 1],dp[i - 2] + nums[i - 1]);
        } 
        return dp[n];
    }

    int rob(vector<int>& nums) {
        if(nums.size() == 1) return nums[0];

        vector<int> nums_;
        nums_.assign(nums.begin() + 1,nums.end());

        nums.pop_back();
        return max(rob1(nums_),rob1(nums));
    }
};
*/


/*
//9.19 T2560 打家劫舍4

class Solution {
public:
    vector<int> arr;
    int findMin(vector<vector<int>>& dp,int idx,int k)
    {
        if(idx >= dp.size() - 1) return 0; //越界啦
        
        if(k > (dp.size() - idx)/2 )
        {
            dp[idx][k] = INT_MAX - 1; //相当于剪枝
            return INT_MAX - 1;
        }
        if(dp[idx][k] != -1) return dp[idx][k]; //已经计算过

        int tmp = findMin(dp,idx + 2,k - 1);
        if(tmp == INT_MAX - 1)
        {
            dp[idx][k] = min(arr[idx],findMin(dp,idx + 1,k));
        }
        else
        {
            dp[idx][k]= min(max(tmp,arr[idx]),findMin(dp,idx + 1,k));
        }

        return dp[idx][k];
    }

    int minCapability(vector<int>& nums, int k) {
        arr = nums; //给参数赋值

        int len = nums.size(),i,j;
        vector<vector<int>> dp(len + 1,vector<int>(k + 1,-1));
        for(i = 0;i <= len;i++) dp[i][0] = 0;
        for(j = 0;j <= k;j++) dp[len][j] = 0;
        
        return findMin(dp,0,k);
    }
};
*/


/*
//9.22 
class Solution {
public:
    int distMoney(int money, int children) {
        if(money < children) return -1;
        int num = money/8;
        if(num >= children) return children;

        int a = money % 8;
        if(a == 4) return num - 1;
        return num;
    }
};
*/


/*
//9.23 T1993 树上的操作

class LockingTree {
public:
    vector<pair<bool,int>> locked; //是否上锁、谁上的锁
    vector<int> fa;  //父节点数组
    vector<vector<int>> son; //子节点数组

    LockingTree(vector<int>& parent) {
        int len = parent.size(); //总节点个数
        fa = parent; //初始化father数组

        vector<pair<bool,int>> locked_(len,pair<bool,int>(false,-2));
        locked = locked_; //初始化locked数组

        son.resize(len);
        for(int i = 1;i < len;i++) son[parent[i]].push_back(i); //初始化son数组
    }

    bool sonIsLocked(int num) //检查是否有子节点被上锁
    {
        if(locked[num].first) return true; //有一个被上锁的
        if(!son[num].empty()) return false; //没有子节点，返回false

        bool flag = false;
        for(int sons : son[num])
        {
            flag = flag || sonIsLocked(sons); //向下递归
        }

        return flag;
    }

    bool faIsUnlocked(int num) //检查是否有父节点被上锁
    {
        if(num == 0) return !locked[0].first; //递归到顶
        if(locked[fa[num]].first) return false; //看父节点有没有上锁

        return faIsUnlocked(fa[num]); //向上递归
    }

    bool lock(int num, int user) {
        if(!locked[num].first) //未上锁
        {
            locked[num] = pair<bool,int>(true,user); //赋值
            return true;
        }

        return false; //已经被上锁
    }
    
    bool unlock(int num, int user) {
        bool flag = (locked[num].first && (locked[num].second == user)); //被该指定用户上锁
        if(flag)
        {
            locked[num] = pair<bool,int>(false,-2); //解锁后回归初始状态
        }

        return flag;
    }

    void upgrading(int num)
    {
        locked[num] = pair<bool,int>(false,-2); //解锁本节点
        if(!son[num].empty()) return; //没有儿子
        for(int i : son[num])
        {
            upgrading(i); //对儿子节点进行操作
        }
    }
    
    bool upgrade(int num, int user) {
        if(!locked[num].first)
        {
            if(sonIsLocked(num) && faIsUnlocked(num)) 
            {    
                upgrading(num); //解锁其子节点
                locked[num] = pair<bool,int>(true,user); //给当前节点上锁
                return true;
            }
        }

        return false;
    }
};
*/


/*
//9.24 T146

class Dchain { //双向链表
public:
    int val;
    Dchain* prev;
    Dchain* next;
    Dchain(int val_ = 0,Dchain* prev_ = nullptr,Dchain* next_ = nullptr):val(val_),prev(prev_),next(next_){} //初始化列表
};

class LRUCache {
public:
    int cap; //容量
    int num; //现在有多少个数
    Dchain* head;//头节点，尾节点
    Dchain* tail; 
    unordered_map<int,Dchain*> map; //哈希表

    LRUCache(int capacity) {
        cap = capacity;
        num = 0;
        head = new Dchain(); //头节点
        tail = new Dchain(); //尾节点
    }

    void moveTotail(Dchain *tmp,bool flag) //把节点放到最后面去，falg = true表示这个点原本在里面，需要先把本体删掉
    {
        if(flag && !tmp)
        {
            tmp->prev->next = tmp->next; //删掉当前节点
            tmp->next->prev = tmp->prev;
        }

        if(num == 0) //没有元素
        {
            tmp->prev = head;
            tmp->next = tail;
            tail->prev = tmp;
            head->next = tmp;
            return;
        }

        else if(!tail)
        {
            tail->prev->next = tmp; //加到末尾
            tmp->prev = tail->prev;
            tmp->next = tail;
            tail->prev = tmp;
            return;
        }
    }
    void takeTheleast() //把头部的关键字拿出去
    {
        if(!head) return;

        head->next = head->next->next;
        head->next->prev = head;
    }
    
    int get(int key) {
        if(map.count(key)) //存在，则移动到最末尾并返回值
        {
            moveTotail(map[key],true);
            return map[key]->val;
        }

        return -1;
    }
    
    void put(int key, int value) {
        if(map.count(key))
        {
            moveTotail(map[key],true); //已访问
            map[key]->val = value; //改变值
            return;
        }

        else
        {
            Dchain* tmp = new Dchain(key);
            moveTotail(tmp,false); //插入尾端
            map.insert({key,tmp});
            num++; //容量+1
            if(num > cap)
            {
                num--;
                takeTheleast(); //删除头部元素
            }
        }
    }
};
*/


/*
//10.1 T121 买卖股票的最佳时机

class Solution { //实时更新最小值，用递归来写
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len == 1) return 0; //特殊情况

        vector<int> res(len);
        res[1] = max(prices[1] - prices[0],0) //赋初值
        int maxi = max(prices[0],prices[1]); //最小值

        for(int i = 2;i < len;i++)
        {
            res[i] = (prices[i] > maxi) ? res[i - 1] + prices[i] - maxi : res[i - 1];
            maxi = max(maxi,prices[i]) //实时更新最小值
        }

        return res[len - 1];
    }
};
*/


/*
//10.5 T309

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size(); //数组长度
        //赋初值，为压缩空间做准备
        int Shares = -prices[0],Shares_; 
        int noShares = 0,noShares_;
        int cool = 0,cool_;

        for(int i = 1;i < len;i++)
        {
            //储存之前的值
            cool_ = cool;Shares_ = Shares;noShares_ = noShares;

            Shares = max(Shares_,noShares_ - prices[i]); //非冷冻期买股票
            noShares = max(noShares_,cool);
            cool = Shares_ + prices[i]; //把股票卖掉啦
        }

        return max(noShares,cool);
    }
};
*/


/*
//10.6 T714

class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int len = prices.size();
        int haveShare = -prices[0],noShare = 0,haveShare_,noShare_; //带下划线的储存上一轮的值

        for(int i = 1;i < len;i++)
        {
            haveShare_ = haveShare;
            noShare_ = noShare;
            noShare = max(noShare_,haveShare_ + prices[i] - fee);
            haveShare = max(haveShare_,noShare_ - prices[i]);
        }

        return noShare;
    }
};
*/


/*
//10.7 T901
class StockSpanner {
public:
    stack<pair<int,int>> stk;
    int num; //数的个数
    StockSpanner() {
        stk.emplace(INT_MAX,-1); //垫底的元素，它不能被出栈
        num = 0; //记录总个数
    }
    
    int next(int price) {
        while(stk.top().first <= price) stk.pop(); //直到找到一个比price大的数
        int res = stk.top().second - num; //要返回的值
        stk.emplace(price,num++); //进入栈中

        return res;
    }
};
*/


/*
//10.8 T2034
class StockPrice {
public:
    unordered_map<int,int> hash; //哈希表对应时间戳与股票价格
    multiset<int> money; //储存所有的当前股票价格，并以O(logn)进行各种查找操作
    int curStamp; // 储存股票最新的价对应的时间戳

    StockPrice() {curStamp = 0;} //初始化
    
    void update(int timestamp, int price) { //如果在，则更新；如果不在，则增加
        auto it1 = hash.find(timestamp);
        curStamp = max(curStamp,timestamp); //更新最新时间戳
        if(it1 == hash.end()) //不在哈希表里面
        {
            hash[timestamp] = price; //哈希表里面增加
            money.emplace(price); //有序集合里面加
        }
        else //在哈希表里面，意味着要更新（注意个数！erase是全部抹掉！所以当个数大的时候，不能全部抹掉）
        {
            int tmp = it1->second; //储存要更新的值
            it1->second = price; //哈希表更新
            //有序集更新
            auto it2 = money.find(tmp);
            money.erase(it2);
            money.emplace(price); 
        }
    }
    
    int current() {return hash[curStamp];}
    
    int maximum() {return *prev(money.end());}
    int minimum() {return *money.begin();}
};
*/


/*
//10.9 T2578
class Solution {
public:
    int splitNum(int num) {
        string s = to_string(num); //按数位排列
        sort(s.begin(),s.end()); //排序
        int len = s.size();
        // 一个一个分配
        int num1 = s[0] - '0',num2 = s[1] - '0';
        for(int i = 2;i < len;i++)
        {
            if(i % 2 == 0)
            {
                num1 = num1 * 10 + s[i] - '0';
            }
            else
            {
                num2 = num2 * 10 + s[i] - '0';
            }
        }

        return num1 + num2;
    }
};
*/


/*
// 10.10 T2731
const int N = 1e9 + 7;
typedef long long LL;

class Solution {
public:
    int sumDistance(vector<int>& nums, string s, int d) {
        int len = nums.size(); //机器人个数
        vector<LL> new_num(len); //要用LL类型存储

        for(int i = 0;i < len;i++) //直接计算不相撞的最后的情况
        {
            if(s[i] == 'R') new_num[i] = nums[i] + d;
            else new_num[i] = nums[i] - d;
        }
        sort(new_num.begin(),new_num.end()); //对数组进行排序，方便计算它的和

        int res = 0; //答案
        for(int i = 1;i < len;i++)
        {
            res += ((new_num[i] - new_num[i - 1]) % N * i * (len - i) % N) % N;
        }

        return res % N;
    }
};
*/


/*
//10.11 T2512

class Solution {
public:
    vector<int> topStudents(vector<string>& positive_feedback, vector<string>& negative_feedback, vector<string>& report, vector<int>& student_id, int k) {
        
    }
};
*/


//10.12 T2562
/*
typedef long long LL;
class Solution {
public:
    long long findTheArrayConcVal(vector<int>& nums) {
        LL res = 0; //答案
        int len = nums.size(); //数组长度
        int i,j;

        for(i = 0,j = len - 1;i <= j;i++,j--)
        {
            if(i == j) res += nums[i];
            else
            {
                string s1 = to_string(nums[i]),s2 = to_string(nums[j]);
                s1 += s2;
                res += stoi(s1);
            }
        }
        return res;
    }
};
*/


/*
//10.12 T1130
class Solution {
public:
    vector<vector<int>> dp; //记忆化搜索

    int f(vector<int>& arr,int left,int right) //left和right是子树的边界
    {
        if(right == left) {dp[left][right] = 0;return 0;} //只有一个数
        if(right - left == 1) {dp[left][right] = arr[right] * arr[left];return dp[left][right];} //间隔中一共只有两个数，直接返回乘积；

        int res = INT_MAX;
        for(int i = left;i < right;i++) //这是根节点左右子树，左边是left-i，右边是i+1-right
        {
            int max1 = 0,max2 = 0; //两侧的最大值
            for(int j = left;j <= i;j++) max1 = max(max1,arr[j]);
            for(int j = i + 1;j <= right;j++) max2 = max(max2,arr[j]); 

            int s1,s2; //储存f(arr,left,i)和f(arr,i + 1,right)
            if(dp[left][i] != -1) s1 = dp[left][i]; //已经访问过
            else //没访问过
            {
                s1 = f(arr,left,i);
                dp[left][i] = s1; //增加访问标志
            }

            if(dp[i + 1][right] != -1) s2 = dp[i + 1][right]; //已经访问过
            else //没访问过
            {
                s2 = f(arr,i + 1,right);
                dp[i + 1][right] = s2;
            }
            
            res = min(res,max1 * max2 + s1 + s2); //动态规划过程
        }

        return res;
    }
    int mctFromLeafValues(vector<int>& arr) {
        int len = arr.size();
        //初始化记忆矩阵
        dp.resize(len);
        for(int i = 0;i < len;i++) dp[i].resize(len,-1);

        return f(arr,0,len - 1);
    }
};
*/


/*
//10.13 T1488

class Solution {
public:
    vector<int> avoidFlood(vector<int>& rains) {
        int len = rains.size(); //数组长度
        vector<int> noRains; //不下雨的天数的序号
        int i;
        for(i = 0;i < len;i++)
        {
            if(rains[i] == 0) noRains.push_back(i);
        }

        set<int> s; //储存一个一个的rains，用来判断
        for(i = 0;i < len - 1;i++)
        {
            int leave = 
            if(s.find(rains[i]) == s.end()) //rains[i]不在s里面
            {
                s.emplace(rains[i]); //插入
                
            }
        }
    }
};
*/


/*
//10.14 T136
// a ^ a = 0,a ^ 0 = a,a ^ b ^ a = b

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = nums[0],len = nums.size();
        for(int i = 1;i < len;i++)
        {
            res ^= nums[i]; //用异或来找，最后剩下的数就是重复两次的数字
        }
        return res;
    }
};
*/


/*
//10.15 T137
class Solution1 { //空间复杂度O(n)
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int,int> hash;
        for(int num : nums) hash[num]++;
        for(int num : nums)
        {
            if(hash[num] == 1) return num;
        }
        return 0;
    }
};

class Solution2 { //空间复杂度O(1)
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for(int i = 0;i < 32;i++) //每一轮的res表示第i位数的和
        {
            int bit = 0; //表示所有数字每一位的和
            for(int num : nums)
            {
                bit += (num >> i) & 1; //num二进制第i位的值
            }
            if(bit % 3 != 0) res += pow(2,i);
        }
        return res;
    }
};
*/


/*
//10.16 T260

class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int c = 0;
        for(int num : nums) c = c ^ num;
        //此时c是a ^ b
        int numOfBit; //c的第一个1所在的位数，也就是a、b在第numOfBit位不相同
        for(numOfBit = 0;numOfBit < 32;numOfBit++) if((c >> numOfBit) & 1) break;    

        int a = 0;
        for(int num : nums)
        {
            if(num >> numOfBit & 1) a = a ^ num; //把第numOfBit位为1的数字归为一类，并算出它们这一类的为唯一的数字
        }

        return {a,a ^ c};
    }
};
*/


/*
//10.17 T2652

class Solution1 {
public:
    int sumOfMultiples(int n) {
        // 容斥原理
        vector<int> arr1 = {3,5,7,105}; // 要加的
        vector<int> arr2 = {15,21,35}; //要减的
        int res = 0;
        for(int n1 : arr1)
        {
            int num = n/n1;
            res += num * (num + 1)/2.0 * n1;
        }
        for(int n2 : arr2)
        {
            int num = n/n2;
            res -= num * (num + 1)/2.0 * n2;
        }
        return res;
    }
};

class Solution2 { //打表
public:
    int sumOfMultiples(int n) {
        int ans[]={0,0,0,3,3,8,14,21,21,30,40,40,52,52,66,81,81,81,99,99,119,140,140,140,164,189,189,216,244,
        244,274,274,274,307,307,342,378,378,378,417,457,457,499,499,499,544,544,544,592,641,691,742,742,742,
        796,851,907,964,964,964,1024,1024,1024,1087,1087,1152,1218,1218,1218,1287,1357,1357,1429,1429,1429,
        1504,1504,1581,1659,1659,1739,1820,1820,1820,1904,1989,1989,2076,2076,2076,2166,2257,2257,2350,2350,
        2445,2541,2541,2639,2738,2838,2838,2940,2940,2940,3045,3045,3045,3153,3153,3263,3374,3486,3486,3600,
        3715,3715,3832,3832,3951,4071,4071,4071,4194,4194,4319,4445,4445,4445,4574,4704,4704,4836,4969,4969,
        5104,5104,5104,5242,5242,5382,5523,5523,5523,5667,5812,5812,5959,5959,5959,6109,6109,6109,6262,6416,
        6571,6727,6727,6727,6886,7046,7207,7369,7369,7369,7534,7534,7534,7702,7702,7872,8043,8043,8043,8217,
        8392,8392,8569,8569,8569,8749,8749,8931,9114,9114,9299,9485,9485,9485,9674,9864,9864,10056,10056,10056,
        10251,10447,10447,10645,10645,10845,11046,11046,11249,11453,11658,11658,11865,11865,11865,12075,12075,
        12075,12288,12288,12503,12719,12936,12936,13155,13375,13375,13597,13597,13821,14046,14046,14046,14274,
        14274,14504,14735,14735,14735,14969,15204,15204,15441,15679,15679,15919,15919,15919,16162,16162,16407,
        16653,16653,16653,16902,17152,17152,17404,17404,17404,17659,17659,17659,17917,18176,18436,18697,18697,
        18697,18961,19226,19492,19759,19759,19759,20029,20029,20029,20302,20302,20577,20853,20853,20853,21132,
        21412,21412,21694,21694,21694,21979,21979,22266,22554,22554,22844,23135,23135,23135,23429,23724,23724,
        24021,24021,24021,24321,24622,24622,24925,24925,25230,25536,25536,25844,26153,26463,26463,26775,26775,
        26775,27090,27090,27090,27408,27408,27728,28049,28371,28371,28695,29020,29020,29347,29347,29676,30006,
        30006,30006,30339,30339,30674,31010,31010,31010,31349,31689,31689,32031,32374,32374,32719,32719,32719,
        33067,33067,33417,33768,33768,33768,34122,34477,34477,34834,34834,34834,35194,35194,35194,35557,35921,
        36286,36652,36652,36652,37021,37391,37762,38134,38134,38134,38509,38509,38509,38887,38887,39267,39648,
        39648,39648,40032,40417,40417,40804,40804,40804,41194,41194,41586,41979,41979,42374,42770,42770,42770,
        43169,43569,43569,43971,43971,43971,44376,44782,44782,45190,45190,45600,46011,46011,46424,46838,47253,
        47253,47670,47670,47670,48090,48090,48090,48513,48513,48938,49364,49791,49791,50220,50650,50650,51082,
        51082,51516,51951,51951,51951,52389,52389,52829,53270,53270,53270,53714,54159,54159,54606,55054,55054,
        55504,55504,55504,55957,55957,56412,56868,56868,56868,57327,57787,57787,58249,58249,58249,58714,58714,
        58714,59182,59651,60121,60592,60592,60592,61066,61541,62017,62494,62494,62494,62974,62974,62974,63457,
        63457,63942,64428,64428,64428,64917,65407,65407,65899,65899,65899,66394,66394,66891,67389,67389,67889,
        68390,68390,68390,68894,69399,69399,69906,69906,69906,70416,70927,70927,71440,71440,71955,72471,72471,
        72989,73508,74028,74028,74550,74550,74550,75075,75075,75075,75603,75603,76133,76664,77196,77196,77730,
        78265,78265,78802,78802,79341,79881,79881,79881,80424,80424,80969,81515,81515,81515,82064,82614,82614,
        83166,83719,83719,84274,84274,84274,84832,84832,85392,85953,85953,85953,86517,87082,87082,87649,87649,
        87649,88219,88219,88219,88792,89366,89941,90517,90517,90517,91096,91676,92257,92839,92839,92839,93424,
        93424,93424,94012,94012,94602,95193,95193,95193,95787,96382,96382,96979,96979,96979,97579,97579,98181,
        98784,98784,99389,99995,99995,99995,100604,101214,101214,101826,101826,101826,102441,103057,103057,
        103675,103675,104295,104916,104916,105539,106163,106788,106788,107415,107415,107415,108045,108045,
        108045,108678,108678,109313,109949,110586,110586,111225,111865,111865,112507,112507,113151,113796,
        113796,113796,114444,114444,115094,115745,115745,115745,116399,117054,117054,117711,118369,118369,
        119029,119029,119029,119692,119692,120357,121023,121023,121023,121692,122362,122362,123034,123034,
        123034,123709,123709,123709,124387,125066,125746,126427,126427,126427,127111,127796,128482,129169,
        129169,129169,129859,129859,129859,130552,130552,131247,131943,131943,131943,132642,133342,133342,
        134044,134044,134044,134749,134749,135456,136164,136164,136874,137585,137585,137585,138299,139014,
        139014,139731,139731,139731,140451,141172,141172,141895,141895,142620,143346,143346,144074,144803,
        145533,145533,146265,146265,146265,147000,147000,147000,147738,147738,148478,149219,149961,149961,
        150705,151450,151450,152197,152197,152946,153696,153696,153696,154449,154449,155204,155960,155960,
        155960,156719,157479,157479,158241,159004,159004,159769,159769,159769,160537,160537,161307,162078,
        162078,162078,162852,163627,163627,164404,164404,164404,165184,165184,165184,165967,166751,167536,
        168322,168322,168322,169111,169901,170692,171484,171484,171484,172279,172279,172279,173077,173077,
        173877,174678,174678,174678,175482,176287,176287,177094,177094,177094,177904,177904,178716,179529,
        179529,180344,181160,181160,181160,181979,182799,182799,183621,183621,183621,184446,185272,185272,
        186100,186100,186930,187761,187761,188594,189428,190263,190263,191100,191100,191100,191940,191940
        191940,192783,192783,193628,194474,195321,195321,196170,197020,197020,197872,197872,198726,199581,
        199581,199581,200439,200439,201299,202160,202160,202160,203024,203889,203889,204756,205624,205624,
        206494,206494,206494,207367,207367,208242,209118,209118,209118,209997,210877,210877,211759,211759,
        211759,212644,212644,212644,213532,214421,215311,216202,216202,216202,217096,217991,218887,219784,
        219784,219784,220684,220684,220684,221587,221587,222492,223398,223398,223398,224307,225217,225217,
        226129,226129,226129,227044,227044,227961,228879,228879,229799,230720,230720,230720,231644,232569,
        232569,233496,233496,233496,234426,235357,235357,236290,236290,237225,238161,238161,239099,240038,
        240978,240978,241920,241920,241920,242865,242865,242865,243813,243813,244763,245714,246666,246666,
        247620,248575,248575,249532,249532,250491,251451,251451,251451,252414,252414,253379,254345,254345,
        254345,255314,256284,256284,257256,258229,258229,259204,259204,259204,260182,260182,261162,262143,
        262143,262143,263127,264112,264112,265099,265099,265099,266089,266089,266089,267082,268076,269071,
        270067,270067,270067,271066,272066};
        return ans[n];
    }
};
*/


/*
//10.18 T2530
typedef long long LL;
class Solution {
public:
    long long maxKelements(vector<int>& nums, int k) {
        LL res = 0;
        int len = nums.size();
        priority_queue<int,vector<int>,less<int>> A; //大顶堆，大的在顶上
        for(int num : nums)
        {
            A.push(num);
        }
        while(k--)
        {
            int large = A.top();
            A.pop();
            res += large;
            A.push(ceilf(large/3.0));
        }
        return res;
    }
};
*/


/*
//10.19 T1726

class Solution {
public:
    int tupleSameProduct(vector<int>& nums) {
        unordered_map<int,int> hash; //储存乘积的哈希表
        int len = nums.size();

        for(int i = 0;i < len;i++)
        {
            for(int j = i + 1;j < len;j++)
            {
                hash[nums[i] * nums[j]]++; //储存乘积
            }
        }

        int res = 0;
        for(auto p : hash)
        {
            int s = p.second;
            res += (s * (s - 1))/2 * 8; //一组相同的贡献八个
        }
        return res;
    }
};
*/


/*
//10.21 T2316

class Solution {
public:
    long long countPairs(int n, vector<vector<int>>& edges) {
        vector<vector<int>> adj(n); //邻接表储存
        vector<bool> visited(n); //判断是否访问过
        long long res = 0;
        for(auto edge : edges)
        {
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }
        
        queue<int> que;
        int top = 0;
        for(int i = 0;i < n;i++)
        {
            if(!visited[i])
            {
                que.push(i);visited[i] = true;
                long long num = 1;
                while(!que.empty())
                {
                    top = que.front();que.pop();
                    for(int dot : adj[top]) //与top相连的点
                    {
                        if(!visited[dot])
                        {
                            que.push(dot);
                            visited[dot] = true;
                            num++;
                        }
                    }
                }
                res += (num * (num - 1)) / 2;
            }
        }
        long long n_ = n;
        return (n_ * (n_ - 1))/2 - res;
    }
};
*/


/*
// 10.25 T2698

class Solution {
public:
    bool dfs(string &s,int start, int tar) //start是开始的位数，tar是目标值
    {
        int len = s.size(); //数组的长度
        if(start == len) return tar == 0;

        for(int j = start;j < len;j++) //分成两组，start---j、j+1---len-1
        {
            string tmp = s.substr(start, j - start + 1); //tmp记录当前的值
            if(dfs(s, j + 1, tar - stoi(tmp))) return true; //dfs搜索
        }
        return false;
    }
    int punishmentNumber(int n) {
        int res = 0;
        for(int i = 1;i <= n;i++)
        {
            string s = to_string(i * i); //转换为字符串，方便操作
            if(dfs(s, 0, i)) res += i * i;
        }
        return res;
    }
};
*/


/*
//10.30 T275
int len;
class Solution {
public:
    int Find(vector<int>& citations, int left, int right)
    {
        if(left > right) return -1;
        int mid = (left + right) / 2; //中间数

        if(citations[mid] == len - mid) return 1;
        if(citations[mid] < len - mid) return Find(citations, mid + 1, right);
        if(citations[mid] > len - mid) return Find(citations, left, mid);
    }

    int hIndex(vector<int>& citations) {
        //寻找a[i]>=n-i
        len = citations.size(); //论文总数

        return Find(citations, 0, len - 1); //后面是界限，用递归求解
    }
};
*/


/*
//2024.7.2 T3115 
class Solution {
public:
    bool isPrime(int num)
    {
        if(num == 1) return false;

        int i = 0;
        for(i = 2; i < num; i++)
        {
            if(num % i == 0) return false;
        }
        return true;
    }

    int maximumPrimeDifference(vector<int>& nums) {
        int n = nums.size();
        int max_idx = 0;
        int min_idx = 0;

        for(int i = 0;i < n; i++)
        {
            if(isPrime(nums[i]))
            {
                min_idx = i;
                break;
            }
        }
        for(int j = n - 1; j >= 0; j--)
        {
            if(isPrime(nums[j]))
            {
                max_idx = j;
                break;
            }
        }

        return max_idx - min_idx;
    }
};
*/


/*
// 2024.7.3 T3099
class Solution {
public:
    int sum(int x)
    {
        string str = to_string(x);
        int res = 0;

        for(int i = 0;i < str.size(); i++) res += str[i] + 0 - '0';

        return res;
    }

    int sumOfTheDigitsOfHarshadNumber(int x) {
        int sum_of_digit = sum(x);
        if(x % sum_of_digit == 0) return sum_of_digit;
        else return -1;
    }
};
*/


/*
//2024.7.14 T807 城市天际线
//给你一座由 n x n 个街区组成的城市，每个街区都包含一座立方体建筑。给你一个下标从 0 开始的 n x n 整数
//矩阵 grid ，其中 grid[r][c] 表示坐落于 r 行 c 列的建筑物的 高度 。
//在 不改变 从任何主要方向观测到的城市 天际线 的前提下，返回建筑物可以增加的 最大高度增量总和 。

class Solution {
public:
    int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
        int n = grid[0].size();\
        int res = 0;
        vector<vector<int>> skyline(2, vector<int>(n, 0));
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
            {
                skyline[0][i] = max(skyline[0][i], grid[i][j]);
                skyline[1][j] = max(skyline[1][j], grid[i][j]);
            }
        }
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
            {
                res += min(skyline[0][i], skyline[1][j]) - grid[i][j];
            }
        }
        return res;
    }
};
*/



/*
//2024.7.24 T2766
// 给你一个下标从 0 开始的整数数组 nums ，表示一些石块的初始位置。再给你两个长度 相等 下标从 0 开始的整数数组 moveFrom 和 moveTo 。
// 在 moveFrom.length 次操作内，你可以改变石块的位置。在第 i 次操作中，你将位置在 moveFrom[i] 的所有石块移到位置 moveTo[i] 。
// 完成这些操作后，请你按升序返回所有 有 石块的位置。如果一个位置至少有一个石块，我们称这个位置 有 石块。一个位置可能会有多个石块。
class Solution {
public:
    vector<int> relocateMarbles(vector<int>& nums, vector<int>& moveFrom, vector<int>& moveTo) {
        unordered_map<int, int> num_index; // num_index[5]表示5位置有多少个石头
        for(int rock_index: nums) num_index[rock_index]++; // 用nums初始化num_index

        for(int i = 0; i < moveFrom.size(); i++) // 模拟过程
        {
            if(moveTo[i] != moveFrom[i])
            {
                num_index[moveTo[i]] += num_index[moveFrom[i]];
                num_index[moveFrom[i]] = 0;
            }
        }
        // 遍历哈希表, 找到答案
        vector<int> res = {};
        unordered_map<int, int>::iterator iter = num_index.begin();
        for(; iter != num_index.end(); iter++)
        {
            if(iter->second != 0) res.push_back(iter->first);
        }
        sort(res.begin(), res.end());

        return res;
    }
};
*/

/*
// T2844
// 给你一个下标从 0 开始的字符串 num ，表示一个非负整数。
// 在一次操作中，您可以选择 num 的任意一位数字并将其删除。请注意，如果你删除 num 中的所有数字，则 num 变为 0。
// 返回最少需要多少次操作可以使 num 变成特殊数字。
// 如果整数 x 能被 25 整除，则该整数 x 被认为是特殊数字。
class Solution {
public:
    int minimumOperations(string num) { // 结尾为00 25 50 75
        
    }
};
*/

/*
//T3106 满足距离约束且字典序最小的字符串
// 给你一个字符串 s 和一个整数 k 。
// 定义函数 distance(s1, s2) ，用于衡量两个长度为 n 的字符串 s1 和 s2 之间的距离，即：
// 字符 'a' 到 'z' 按 循环 顺序排列，对于区间 [0, n - 1] 中的 i ，计算所有「 s1[i] 和 s2[i] 之间 最小距离」的 和 。
// 例如，distance("ab", "cd") == 4 ，且 distance("a", "z") == 1 。
// 你可以对字符串 s 执行 任意次 操作。在每次操作中，可以将 s 中的一个字母 改变 为 任意 其他小写英文字母。
// 返回一个字符串，表示在执行一些操作后你可以得到的 字典序最小 的字符串 t ，且满足 distance(s, t) <= k 
class Solution {
public:
    string getSmallestString(string s, int k) {
        int num = s.size();
        for(int i = 0; i < num; i++)
        {
            if((s[i] - 'a' <= k) || ('z' - s[i] + 1 <= k)) // 判断是否可以转化到'a'，第一个是向前，第二个是向后
            {
                k = k - min('z' - s[i] + 1, s[i] - 'a'); // 更改消耗的k
                s[i] = 'a';
                if(k == 0) return s; // 额度'k'用完，已经无法继续，提前结束
            }
            else  //如果向后向前都不能转化到a，那么就不能向后转化，应该尽力向前转化
            {
                s[i] = s[i] - k;
                return s; //这一次必然将额度k用完，直接返回
            }
        }
        return s; //如果没用完额度，那么直接返回s
    }
};
*/


/*
// T699 掉落的方块
// 在二维平面上的 x 轴上，放置着一些方块。
// 给你一个二维整数数组 positions ，其中 positions[i] = [lefti, sideLengthi] 表示：第 i 个方块边长为 sideLengthi ，其左侧边与 x 轴上坐标点 lefti 对齐。
// 每个方块都从一个比目前所有的落地方块更高的高度掉落而下。方块沿 y 轴负方向下落，直到着陆到 另一个正方形的顶边 或者是 x 轴上 。
// 一个方块仅仅是擦过另一个方块的左侧边或右侧边不算着陆。一旦着陆，它就会固定在原地，无法移动。
// 在每个方块掉落后，你必须记录目前所有已经落稳的 方块堆叠的最高高度 。
// 返回一个整数数组 ans ，其中 ans[i] 表示在第 i 块方块掉落后堆叠的最高高度。
class Solution {
public:
    vector<int> fallingSquares(vector<vector<int>>& positions) {
        vector<int> res;
        vector<vector<int>> interval; //记录已经落下的区间
        for(int i = 0; i < positions.size(); i++)
        {
            int ans = 0; //这一层的答案
            for(int j = 0; j < interval.size(); j++)
            {
                if((positions[i][0] >= interval[j][1]) || (positions[i][0] +  positions[i][1] <= interval[j][0])) //没有重叠
                {
                    ans = max(ans, positions[i][1]);
                }
                else //有重叠
                {
                    
                }
            }
            res.push_back(ans);
        }
        return res;
    }
};
*/


/*
// 2024.8.5 T600
// 给定一个正整数 n ，请你统计在 [0, n] 范围的非负整数中，有多少个整数的二进制表示中不存在连续的 1 。
class Solution {
public:
    vector<int> res;
    Solution() { res = vector<int>(40, 0); } //初始化函数

    int findIntegersInOne(int n) { //有n个1的数字的答案
        if(n == 0 || n == 1) return 0;
        if(n == 2) return 1;
        if(res[n] != 0) return res[n]; //记忆化搜索

        res[n] = findIntegersInOne(n - 1) + findIntegersInOne(n - 2) + pow(2, n - 2);
        return res[n];
    }

    string toBinary(int n) { //十进制转二进制
        string r;
        while (n != 0) 
        {
            r += (n % 2 == 0 ? "0" : "1");
            n /= 2;
        }
        reverse(r.begin(), r.end());
        return r;
    }

    int toDecimal(string binary) { //二进制转回十进制
        int decimal = 0;
        int length = binary.size();
        for (int i = 0; i < length; ++i) {
            if (binary[length - i - 1] == '1') {
                decimal += pow(2, i);
            }
        }
        return decimal;
    }

    int findIntegers_inverse(int n) { //计算有多少个有连续两个1的
        string bin_n = toBinary(n); 
        bool flag = false; //记录上一位是不是1
        int res = 0; //答案
        for(int i = 0; i < bin_n.size(); i++)
        {
            if(bin_n[i] == '1')
            {
                res += findIntegersInOne(bin_n.size() - i - 1);
                if(flag) { //说明上一位也是1 
                    return res + toDecimal(bin_n.substr(i + 1, bin_n.size() - i)) + 1;
                }
                flag = true; //这一位是1
            }
            else flag = false; //这一位是0
        }
        return res;
    }

    int findIntegers(int n) { return n + 1 - findIntegers_inverse(n);}
};

int main()
{
    Solution A;
    cout << A.findIntegers(27) << endl;
    return 0;
}
*/


/*
//2024.8.6 T3129、3130
// 给你 3 个正整数 zero ，one 和 limit 。
// 一个 二进制数组 arr 如果满足以下条件，那么我们称它是 稳定的 ：
// 0 在 arr 中出现次数 恰好 为 zero 。1 在 arr 中出现次数 恰好 为 one 。
// arr 中每个长度超过 limit 的 子数组都 同时 包含 0 和 1 。
// 请你返回 稳定 二进制数组的 总 数目。
// 由于答案可能很大，将它对 10^9 + 7 取余 后返回。 1 <= zero, one, limit <= 200 （T3130是1000）

typedef long long LL;
const LL MAX = 1e9 + 7;
class Solution {
public:
    vector<vector<LL>> dp_zero, dp_one;
    int lim;
    int size = 1002;
    int numberOfStableArrays(int zero, int one, int limit) {
        lim = limit;
        dp_zero = vector<vector<LL>>(size, vector<LL>(size, -1)); 
        dp_one = vector<vector<LL>>(size, vector<LL>(size, -1)); // 初步初始化
        dp_zero[0][0] = 0; dp_one[0][0] = 0;
        for(int i = 1; i < size; i++) dp_zero[i][0] = (i <= limit) ? 1 : 0; // 初始化 dp_zero和dp_one
        for(int j = 1; j < size; j++) dp_one[0][j] = (j <= limit) ? 1 : 0; 
        
        return (arr0(zero, one) + arr1(zero, one) + MAX) % MAX;
    }

    int arr0(int zero, int one); // 提前声明函数原型
    int arr1(int zero, int one); // 提前声明函数原型
};

int Solution::arr0(int zero, int one) {
    if(zero < 0 || one < 0) return 0;
    else if(dp_zero[zero][one] != -1) return dp_zero[zero][one]; // 记忆化

    dp_zero[zero][one] = (arr0(zero - 1, one) + arr1(zero - 1, one) - arr1(zero - 1 - lim, one) + MAX) % MAX;
    return dp_zero[zero][one];
}

int Solution::arr1(int zero, int one) {
    if(zero < 0 || one < 0) return 0;
    else if(dp_one[zero][one] != -1) return dp_one[zero][one]; // 记忆化

    dp_one[zero][one] = (arr0(zero, one - 1) + arr1(zero, one - 1) - arr0(zero, one - 1 - lim) + MAX) % MAX;
    return dp_one[zero][one];
}
int main()
{   
    Solution A;
    cout << A.numberOfStableArrays(3, 3, 2) << endl;
    return 0;
}
*/



/*
// 2024.12.26
// 3083. 字符串及其反转中是否存在同一子字符串
// 给你一个字符串 s ，请你判断字符串 s 是否存在一个长度为 2 的子字符串，在其反转后的字符串中也出现。
// 如果存在这样的子字符串，返回 true；如果不存在，返回 false 。

class Solution {
public:
    bool isSubstringPresent(string s) {
        int len = s.size();
        unordered_map<string, int> str_map; // 用哈希表记忆长度为 2 的子字符串的个数
        for(int i = 0; i < len - 1; i++)
        {
            if(s[i] == s[i + 1]) return true;
            str_map[s.substr(i, 2)]++;
        }
        reverse(s.begin(), s.end());
        for(int i = 0; i < len - 1; i++)
        {
            if(str_map[s.substr(i, 2)] >= 1) return true;
        }
        return false;
    }
};
*/


/*
// 2025.1.28 
// 119. 杨辉三角 II
// 给定一个非负索引 rowIndex，返回「杨辉三角」的第 rowIndex 行。
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        vector<int> res(rowIndex + 1, 1); // 初始化答案
        int midIndex = rowIndex / 2; // 中间数
        for(int i = 1; i <= midIndex; i++)
        {
            res[i] = 1LL * res[i - 1] * (rowIndex + 1 - i) / i;
            res[rowIndex - i] = res[i];
        }

        return res;
    }
};
*/



