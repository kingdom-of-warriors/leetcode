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
class Solution {//经典拓扑排序
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


//10.11 T2512

class Solution {
public:
    vector<int> topStudents(vector<string>& positive_feedback, vector<string>& negative_feedback, vector<string>& report, vector<int>& student_id, int k) {
        
    }
};
