#include <bits/stdc++.h>

using namespace std;

/*
//7.30 剑指offer T09 用两个栈实现队列
//用两个栈实现一个队列。
//队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
//(若队列中没有元素，deleteHead 操作返回 -1 )

class CQueue {
public:
    stack<int> A,B;
    CQueue() {}

    void appendTail(int value) {
        A.push(value);
    }

    int deleteHead() {
        if(!B.empty())
        {
            int tmp =  B.top();
            B.pop();
            return tmp;
        }
        else if(A.empty())
        {
            return -1;
        }
        else
        {
            while(!A.empty())
            {
                B.push(A.top());
                A.pop();
            }

            int tmp = B.top();
            B.pop();

            return tmp;
        }
    }
};
*/

/*
//7.30 剑指offer T30 包含min函数的栈
//定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

class MinStack {
public:
    stack<int> A,B; //B中储存A的非严格降序序列，可以理解为前i项的最小值
    MinStack() {}

    void push(int x) {
        if(A.empty())
        {
            A.push(x);
            B.push(x);
        }
        else
        {
            if(B.top() >= x) {B.push(x);}
            A.push(x);
        }
    }

    void pop() {
        if(A.top() == B.top()) {B.pop();}
        A.pop();
    }

    int top() {
        return A.top();
    }

    int min() {
        return B.top();
    }
};
*/

/*
//7.30 剑指offer T06 从尾到头打印链表
//输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

//辅助栈方法
struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};

class Solution1 {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> stk;//用于倒序输出
        vector<int> arr;

        ListNode *p = head;
        if(head == nullptr) return arr;
        else
        {
            while(p)
            {
                stk.push(p->val);
                p = p->next;
            }
            while(!stk.empty())
            {
                arr.push_back(stk.top());
                stk.pop();
            }

            return arr;
        }
    }
};

//递归法
class Solution2 {
public:
    vector<int> arr;

    void reverse(ListNode* head)
    {
        if(head)
        {
            reverse(head->next);
            arr.push_back(head->val);
        }
        else return;
    }

    vector<int> reversePrint(ListNode* head) {
        reverse(head);
        return arr;
    }
};
*/

/*
//7.31 剑指offer T24 反转链表
//定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
//输入: 1->2->3->4->5->NULL
//输出: 5->4->3->2->1->NULL

struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        else
        {
            ListNode* p = head->next;
            ListNode* q = p->next;
            head->next = nullptr;
            while(p)
            {
                p->next = head;
                head = p;
                p = q;
                if(q) {q = q->next;}
            }

            return head;
        }
    }
};
*/

/*
//7.31 剑指offer T35 复杂链表的复制
//请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，
//还有一个 random 指针指向链表中的任意节点或者 null。

class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

//哈希表
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr) return head;

        Node* p = head;
        unordered_map<Node*,Node*> map;

        while(p)
        {
            map[p] = new Node(p->val);
            p = p->next;
        }

        p = head;
        while(p)
        {
            map[p]->next = map[p->next];
            map[p]->random = map[p->random];
            p = p->next;
        }

        return map[head];
    }
};
*/

/*
//7.31 剑指offer T05 替换空格
//请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
//示例 1：
//输入：s = "We are happy."
//输出："We%20are%20happy."

class Solution1 {
public:
    string replaceSpace(string s) {
        int n = s.length();
        int i = 0,j = 0;

        int count = 0;
        for(i = 0;i < n;i++)
        {
            if(s[i] == ' ') count++;
        }

        string s1(n + 2*count, '0');

        for(i = 0,j = 0;i < n;i++,j++)
        {
            if(s[i] != ' ') s1[j] = s[i];
            else
            {
                s1[j++] = '%';s1[j++] = '2';s1[j] = '0';
            }
        }
        return s1;
    }
};

//从后往前，空间复杂度O(1)
class Solution2 {
public:
    string replaceSpace(string s) {
        int n = s.size();
        int i = 0,j = 0;

        int count = 0;
        for(i = 0;i < n;i++)
        {
            if(s[i] == ' ') count++;
        }

        s.resize(n + 2*count);
        for(i = n - 1,j = s.size() - 1;i < j; i--,j--)
        {
            if(s[i] != ' ') s[j] = s[i];
            else
            {
                s[j--] = '0';
                s[j--] = '2';
                s[j] = '%';
            }
        }
        return s;
    }
};

int main()
{
    Solution2 A;
    string s = "We are happy.";
    cout<<A.replaceSpace(s);
}
*/

/*
// 7.31 剑指offer T58 左旋转字符串
// 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
// 比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
//  示例 1：
//  输入: s = "abcdefg", k = 2
//  输出: "cdefgab"

class Solution1 {
public:
    string reverseLeftWords(string s, int n) {
        int len = s.size();
        int i = 0;

        string s1(s);
        for(i = 0;i < len;i++)
        {
            s1[i] = s[(i + n) % len];
        }

        return s1;
    }
};

class Solution2 {
public:
    string reverseLeftWords(string s, int n) {
        return s.substr(n,s.size()) + s.substr(0,n);
    }
};


int main()
{
    Solution2 A;
    string s = "abcdefg";
    cout<<A.reverseLeftWords(s,2);
}
*/

/*
//8.1 剑指offer T03 数组中的重复数字
// 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。
// 数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
// 示例 1：
// 输入：[2, 3, 1, 0, 2, 5, 3]
// 输出：2 或 3

// 不用哈希表
const int MaxSize = 100005;
class Solution1 {
public:
    int findRepeatNumber(vector<int>& nums) {
        int len = nums.size();
        int i = 0;
        vector<int> arr(MaxSize);

        for(i = 0;;i++)
        {
            arr[nums[i]]++;
            if(arr[nums[i]] > 1) return nums[i];
        }

        return 0;
    }
};

//哈希表写法
class Solution2 {
public:
    int findRepeatNumber(vector<int>& nums) {
        unordered_map<int,bool> map;
        for(int num:nums)
        {
            if(map[num]) return num;
            map[num] = true;
        }
        return 0;
    }
};


int main()
{
    Solution2 A;
    vector<int> nums = {2,3,1,0,2,5,3};
    cout<<A.findRepeatNumber(nums);
}
/*


/*
//8.1 剑指offer T53 在排序数组中查找数字I
// 统计一个数字在排序数组中出现的次数。
// 示例 1:
// 输入: nums = [5,7,7,8,8,10], target = 8
// 输出: 2

//哈希表写法
class Solution1 {
public:
    int search(vector<int>& nums, int target) {
        unordered_map<int,int> map;

        for(int num:nums)
        {
            map[num]++;
        }

        return map[target];
    }
};

//二分法写法（因为是排序数组），搜索左边界和右边界,左开右闭区间
class Solution2 {
public:
    int search(vector<int>& nums, int target) {
        int left = 0,right1 = nums.size() - 1;
        if(right1 < 0 || nums[right1] < target || nums[0] > target) return 0;

        while(left < right1)
        {
            int mid = (right1 - left)/2 + left;

            if(nums[mid] >= target) right1 = mid;
            else left = mid + 1;
        }

        int left1 = 0,right = nums.size();
        while(left1 < right)
        {
            int mid = (right - left1)/2 + left1;

            if(nums[mid] <= target) left1 = mid + 1;
            else if(nums[mid] > target) right = mid;
        }

        return right - left;
    }
};

int main()
{
    Solution1 A;
    Solution2 B;
    vector<int> nums = {};
    cout<<A.search(nums,0)<<endl;
    cout<<B.search(nums,0)<<endl;
}
*/

/*
//8.2 剑指offer T53 0 ~ n-1中的缺失数字
// 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。
// 在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
// 示例 1:
// 输入: [0,1,3]
// 输出: 2

class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int left = 0,right = nums.size();
        while(left < right)
        {
            int mid = (right - left)/2 + left;
            if(nums[mid] == mid) {left = mid + 1;}
            if(nums[mid] == mid + 1) {right = mid;}
        }

        return left;
    }
};

int main()
{
    Solution A;
    vector<int> nums= {0,1,2,3,4,5,6,7,9};
    cout << A.missingNumber(nums);
}
*/

/*
//8.2 剑指offer T04 二维数组中的查找
//在一个 n * m 的二维数组中，每一行都按照从左到右非递减的顺序排序，每一列都按照从上到下非递减的顺序排序。
//请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i = matrix.size() - 1, j = 0;
        while(i >= 0 && j < matrix[0].size())
        {
            if(matrix[i][j] > target) i--;
            else if(matrix[i][j] < target) j++;
            else return true;
        }
        return false;
    }
};

int main()
{
    Solution A;
    vector<vector<int>> matrix = {{1, 4, 5},
                                  {2, 6, 7},
                                  {3, 9,11},
                                 };
    cout<<A.findNumberIn2DArray(matrix,9);
}
*/

/*
//8.2 剑指offer T11 旋转数组的最小数字
//把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
//给你一个可能存在 重复 元素值的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。
//请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为 1。  
//注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

class Solution {
public:
    int minArray(vector<int>& numbers) {
        int i = 0,j = numbers.size() - 1;

        while(i < j)
        {
            int mid = (j - i)/2 + i;
            if(numbers[mid] > numbers[j]) {i = mid + 1;}
            else if(numbers[mid] < numbers[j]) {j = mid;}
            else {j = j - 1;}
        }

        return numbers[j];
    }
};

int main()
{
    Solution A;
    vector<int> numbers = {2,2};
    cout<<A.minArray(numbers);
}
*/

/*
//8.3 剑指offer T32 从上到下打印二叉树 1
// 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
// 给定二叉树: [3,9,20,null,null,15,7],
//     3
//    / \
//   9  20
//     /  \
//    15   7
// 返回：
// [3,9,20,15,7]

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };

class Solution1 {
public:
    vector<int> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        TreeNode* p = nullptr;
        vector<int> arr;

        if(!root) return arr;
        que.push(root);
        while(!que.empty())
        {
            p = que.front();
            arr.push_back(p->val);
            que.pop();

            if(p->left) {que.push(p->left);}
            if(p->right) {que.push(p->right);}
        }

        return arr;
    }
};
*/

/*
//8.2 剑指offer T32 从上到下打印你二叉树 2 & 3
// 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
//     3       给定二叉树
//    / \
//   9  20
//     /  \
//    15   7
// 返回其层次遍历结果：
// [
//   [3],
//   [9,20],
//   [15,7]
// ]

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };

class Solution2 {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q1;
        vector<vector<int>> res;
        TreeNode* p = nullptr;
        int depth = 0,idx = 0;

        if(!root) return res;

        q1.push(root);

        while(!q1.empty())
        {
            vector<int> tmp;
            for(int i = q1.size();i > 0;i--)
            {
                p = q1.front();
                q1.pop();
                tmp.push_back(p->val);

                if(p->left) {q1.push(p->left);}
                if(p->right) {q1.push(p->right);}
            }
            res.push_back(tmp);
        }
        return res;
    }
};

class Solution3 {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q1;
        vector<vector<int>> res;
        if(!root) return res;

        TreeNode* p = nullptr;
        int cnt = 0;
        q1.push(root);

        while(!q1.empty())
        {
            cnt++;
            vector<int> tmp;
            for(int i = q1.size();i > 0;i--)
            {
                p = q1.front();
                q1.pop();
                tmp.push_back(p->val);

                if(p->left) {q1.push(p->left);}
                if(p->right) {q1.push(p->right);}
            }
            if(cnt % 2 == 0) reverse(tmp.begin(),tmp.end());
            res.push_back(tmp);
        }
        return res;
    }
};
*/

/*
//8.2 剑指offer T26 树的子结构
//输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
//B是A的子结构， 即 A中有出现和B相同的结构和节点值。


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };

class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(!A || !B) return false;
        return recur(A,B) || isSubStructure(A->left,B) || isSubStructure(A->right,B);
    }

    bool recur(TreeNode* A,TreeNode* B)
    {
        if(!B) return true;
        if(!A || A->val != B->val) return false;
        return recur(A->left,B->left) && recur(A->right,B->right);
    }
};
*/

/*
//8.3 剑指offer T27 二叉树镜像
// 请完成一个函数，输入一个二叉树，该函数输出它的镜像。例如输入：
//      4
//    /   \
//   2     7
//  / \   / \
// 1   3 6   9
// 镜像输出：
//      4
//    /   \
//   7     2
//  / \   / \
// 9   6 3   1

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(!root) return root;

        swap(root->left,root->right);
        root->left = mirrorTree(root->left);
        root->right = mirrorTree(root->right);

        return root;
    }
};
*/

/*
//8.3 剑指offer T28 对称的二叉树
// 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
// 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
//     1
//    / \
//   2   2
//  / \ / \
// 3  4 4  3

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;

        return f(root,root);
    }

    bool f(TreeNode* A,TreeNode* B) {
        if(!A || !B) return !A && !B;  // A B 有一个是空指针 当且仅当两个都是空指针的时候返回
        return (A->val == B->val) && f(A->left,B->right) && f(A->right,B->left);
    }
};
*/

/*
//8.3 剑指offer T63 股票的最大利润
// 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
// 输入: [7,1,5,3,6,4]  输出: 5
// 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
//      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。

//可压缩空间
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        int i = 0;

        if(len == 0 || len == 1) {return 0;}

        vector<int> dp(len);
        int mini = min(prices[0],prices[1]);
        dp[0] = 0;dp[1] = prices[1] - prices[0];

        for(i = 2;i < len;i++)
        {
            dp[i] = max(dp[i - 1],prices[i] - mini);
            mini = min(mini,prices[i]);
        }

        return max(0,dp[len - 1]);
    }
};

int main()
{
    Solution A;
    vector<int> prices = {7};
    cout<<A.maxProfit(prices);
}
*/

/*
//8.4 剑指offer T42 连续子数组最大和
// 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
// 示例1:
// 输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
// 输出: 6
// 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

class Solution1 {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return nums[0];

        int i = 0;
        vector<int> dp(len);
        dp[0] = nums[0];
        int maxi = dp[0];

        for(i = 1;i < len;i++)
        {
            dp[i] = max(dp[i - 1],0) + nums[i];
            maxi = max(maxi,dp[i]);
        }

        return maxi;
    }
};

class Solution2 {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return nums[0];

        int i = 0,a = nums[0],b = 0,maxi = nums[0];

        for(i = 1;i < len;i++)
        {
            b = a;
            a = max(b,0) + nums[i];
            maxi = max(a,maxi);
        }

        return maxi;
    }
};

int main()
{
    Solution2 A;
    vector<int> nums = {-2,1,-3,4,-1,2,1,-5,4};
    cout<<A.maxSubArray(nums);
}
*/

/*
//8.3 剑指offer T47 礼物的最大价值
// 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，
// 并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
// 输入:
// [
//   [1,3,1],
//   [1,5,1],
//   [4,2,1]
// ]
// 输出: 12
// 解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物

class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int row = grid.size();int col = grid[0].size();
        int i = 0,j = 0;

        //在原二维数组上进行更改，赋初值
        for(i = 1;i < row;i++)
        {
            grid[i][0] += grid[i - 1][0];
        }

        for(j = 1;j < col;j++)
        {
            grid[0][j] += grid[0][j - 1];
        }

        for(i = 1;i < row;i++)
        {
            for(j = 1;j < col;j++)
            {
                grid[i][j] += max(grid[i - 1][j],grid[i][j - 1]);
            }
        }

        return grid[row - 1][col - 1];
    }
};
*/

/*
//8.4 剑指offer T46 把数字翻译为字符串
// 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
// 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
// 输入: 12258
// 输出: 5
// 解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"

class Solution {
public:
    int translateNum(int num) {
        if(num <= 9) return 1;
        else if(num <= 25) return 2;
        else if(num <= 99) return 1;

        int tmp1 = num/10, tmp2 = num/100, bit2 = num % 100;

        if(bit2 <= 25 && bit2 >= 10) return translateNum(tmp1) + translateNum(tmp2);
        else return translateNum(tmp1);
    }
};

int main()
{
    Solution A;
    int num = 12211;
    cout<<A.translateNum(num);
}
*/

/*
//8.4 剑指offer T48 最长不含重复字符的子字符串
// 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
// 输入: "abcabcbb"
// 输出: 3
// 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = s.size();
        if(len == 0) return 0;

        int i = 0,j = 0;
        int maxi = 1;
        vector<int> dp(len,1);

        for(i = 1;i < len;i++)
        {
            for(j = i - 1;j >= i - dp[i - 1];j--)
            {
                if(s[j] != s[i]) dp[i]++;
                else break;
            }
            maxi = max(maxi,dp[i]);
        }

        return maxi;
    }
};

int main()
{
    Solution A;
    string s = "abcabcbb";
    cout<<A.lengthOfLongestSubstring(s);
}
*/

/*
//8.4 剑指offer T18 删除链表节点
// 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点。
// 输入: head = [4,5,1,9], val = 5
// 输出: [4,1,9]
// 解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(!head || !head->next) return NULL;

        if(head->val == val)
        {
            head = head->next;
            return head;
        }

        ListNode *p = head->next, *q = head;

        //寻找过程
        while(p->val != val)
        {
            q = p;
            p = p->next;
        }

        q->next = q->next->next;

        return head;
    }
};
*/

/*
//8.5 剑指offer T22 链表中倒数第k个节点
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};


class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *p = head, *q = head;

        while(k > 0)
        {
            k--;
            p = p->next;
        }

        while(p)
        {
            p = p->next;
            q = q->next;
        }

        return q;
    }
};
*/

/*
//8.5 剑指offer T25 合并两个排序的链表
//输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode *p = l1, *q = l2;
        ListNode *head = new ListNode(0);
        ListNode *res = head;

        while(p && q)
        {
            ListNode *tmp = new ListNode(0);

            if(p->val <= q->val)
            {
                tmp->val = p->val;
                p = p->next;
            }
            else
            {
                tmp->val = q->val;
                q = q->next;
            }
            head->next = tmp;
            head = head->next;
        }

        if(!p)
        {
            while(q)
            {
                ListNode *tmp = new ListNode(q->val);
                head->next = tmp;
                head = head->next;

                q = q->next;
            }
        }

        if(!q)
        {
            while(p)
            {
                ListNode *tmp = new ListNode(p->val);
                head->next = tmp;
                head = head->next;

            p = p->next;
            }
        }

        return res;
    }
};
*/

/*
//8.5 剑指offer T52 两个链表的第一个公共节点
//输入两个链表，找出它们的第一个公共节点。

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int lenA = 0,lenB = 0;
        ListNode *A = headA, *B = headB;

        while(A)
        {
            A = A->next;
            lenA++;
        }
        while(B)
        {
            B = B->next;
            lenB++;
        }

        if(lenA >= lenB)
        {
            int i = lenA - lenB;
            while(i--)
            {
                headA = headA->next;
            }
        }
        else
        {
            int i = lenB - lenA;
            while(i--)
            {
                headB = headB->next;
            }
        }

        while(headA != headB)
        {
            headA = headA->next;
            headB = headB->next;
        }
        return headA;
    }
};
*/

/*
//8.5 剑指offer T57 和为s的两个数字
// 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
// 输入：nums = [2,7,11,15], target = 9
// 输出：[2,7] 或者 [7,2]

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res(2);
        int i = 0,j = nums.size() - 1;

        while(i < j)
        {
            if(nums[i] + nums[j] > target) j--;
            else if(nums[i] + nums[j] < target) i++;
            else
            {
                res[0] = nums[i];res[1] = nums[j];
                break;
            }
        }

        return res;
    }
};

int main()
{
    vector<int> arr = {1,3,5,7,9};
    Solution A;
    vector<int> a = A.twoSum(arr,10);
    cout<<a[0]<<" "<<a[1];
}
*/

/*
//8.6 剑指offer T58 反转单词顺序 1
// 一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
// 为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
// 示例 1：
// 输入: "the sky is blue"
// 输出: "blue is sky the"

class Solution {
public:
    string reverseWords(string s) {
        if(s.size() == 0) return "";

        int left = 0,right = s.size() - 1;
        while(s[left] == ' ') left++;
        while(right >= 0 && s[right] == ' ') right--;

        if(right < 0) return "";

        string res;
        queue<string> que;
        int i = right,j = right;

         //分割出一个单词
        while(i >= left && i <= j)
        {
            string word;
            while(i >= left && s[i] != ' ')
            {
                i--;
            }
            word = s.substr(i + 1, j - i);
            que.push(word);

            while(i >= left && s[i] == ' ') i--;

            j = i;
        }

        while(!que.empty())
        {
            res += que.front();
            que.pop();
            if (!que.empty()) res += ' ';
        }

        return res;
    }
};

int main()
{
    Solution A;
    string s = " ";
    cout<<A.reverseWords(s);
}
*/

/*
//8.6 剑指offer T12 矩阵中的路径
// 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
// 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
// 同一个单元格内的字母不允许被重复使用。

class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int row = board.size(),col = board[0].size();

        bool flag = false;
        for(int i = 0;i < row;i++)
        {
            for(int j = 0;j < col;j++)
            {
                if(dfs(board,i,j,word,0)) return true;
            }
        }
        return false;
    }

    bool dfs(vector<vector<char>>& board,int i,int j,string word,int k)
    {
        int row = board.size(),col = board[0].size();

        if(i < 0 || i >= row || j < 0 || j >= col || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';

        bool res = dfs(board,i + 1,j,word,k + 1) || dfs(board,i,j + 1,word,k + 1) || dfs(board,i - 1,j,word,k + 1) || dfs(board,i,j - 1,word,k + 1);
        board[i][j] = word[k];

        return res;
    }
};

int main()
{
    Solution A;
    vector<vector<char>> board = {{'a','b'},{'c','d'}};
    cout<<A.exist(board,"a");
}
*/

/*
//8.7 剑指offer T21 调整数组顺序使奇数位于偶数前面
// 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。
// 输入：nums = [1,2,3,4]
// 输出：[1,3,2,4]
// 注：[3,1,2,4] 也是正确的答案之一。

class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        if(nums.size() == 0 || nums.size() == 1) return nums;

        int left = 0, right = nums.size() - 1;
        while(left < right)
        {
            if(nums[left] % 2 == 1)
            {
                left++;
            }
            else if(nums[right] % 2 == 0)
            {
                right--;
            }
            else
            {
                swap(nums[left],nums[right]);
                left++;
                right--;
            }
        }

        return nums;
    }
};

int main()
{
    Solution A;
    vector<int> nums = {11,9,3,7,16,4,2,0};
    vector<int> arr = A.exchange(nums);
    for(int i = 0;i < nums.size();i++)
    {
        cout<<arr[i]<<' ';
    }
}
*/

/*
//8.7 剑指offer T13 机器人的运动范围
// 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格
// （不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，
// 因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

class Solution {
public:

    int movingCount(int m, int n, int k) {
        vector<vector<int>> mat(m,vector<int>(n,0));//没访问0,访问了且可以为1,访问量且不可以为2
        mat[0][0] = 1;
        int cnt = 1;
        int i = 0,j = 0;

        for(i = 0;i < m;i++)
        {
            for(j = 0;j < n;j++)
            {
                if(i == 0 && j == 0) continue;
                if(canAchieve(i,j,k,m,n,mat)) cnt++;
            }
        }

        return cnt;
    }

    bool canAchieve(int i,int j,int k,int m,int n,vector<vector<int>>& mat)
    {
        if(i >= m || j >= n || i < 0 || j < 0) return false; //越界

        if(mat[i][j] != 0) return (mat[i][j] == 1);//访问过

        if(k < IntSum(i) + IntSum(j))
        {
            mat[i][j] = 2;
            return false;
        }

        bool flag1 = canAchieve(i - 1,j,k,m,n,mat);
        bool flag2 = canAchieve(i,j - 1,k,m,n,mat);

        bool flag = flag1 || flag2;
        if(flag) mat[i][j] = 1;
        else mat[i][j] = 2;

        return flag;
    }

    int IntSum(int num)
    {
        if(num < 10) return num;

        return IntSum(num / 10) + num % 10;
    }
};

int main()
{
    Solution A;
    cout<<A.movingCount(16,8,4);
}
*/

/*
//8.7 剑指offer T34 二叉树中和为某一值的路径
// 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
// 叶子节点 是指没有子节点的节点。

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
    vector<vector<int>> res;
    vector<int> path; //储存当前路径

    vector<vector<int>> pathSum(TreeNode* root, int target) {
        if(!root) return res;

        recur(root,target);
        return res;
    }

    void recur(TreeNode* root,int target)
    {
        if(!root) return;

        path.push_back(root->val);
        target -= root->val;

        if(!root->left && !root->right && target == 0)
        {
            res.push_back(path);
        }

        recur(root->left,target);
        recur(root->right,target);

        path.pop_back();
    }
};
*/

/*
//8.8 剑指offer T36 二叉搜索树与双向链表
//输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}
    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

class Solution1 {
public:
    vector<Node*> line;
    Node* treeToDoublyList(Node* root) {
        if(!root) return;

        traversal(root);

        for(int i = 0;i < line.size();i++)
        {
            line[i]->right = line[(i + 1) % line.size()];
            line[(i + 1) % line.size()]->left = line[i];
        }

        return line[0];
    }

    void traversal(Node* root)
    {
        if(!root) return;

        traversal(root->left);
        line.push_back(root);
        traversal(root->right);
    }
};
*/

/*
//8.9 剑指offer T54 二叉搜索树的第k大节点
//给定一棵二叉搜索树，请找出其中第 k 大的节点的值。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        stack<TreeNode*> stk1;
        stack<int> stk2;
        int cnt = 0,flag = 0;
        TreeNode* p = nullptr;

        stk1.push(root);
        stk2.push(0);

        while(!stk1.empty())
        {
            flag = stk2.top();stk2.pop();
            p = stk1.top();

            switch (flag)
            {
                case 0:
                {
                    stk2.push(1);
                    if(p->right)
                    {
                        stk1.push(p->right);stk2.push(0);
                    }
                    break;
                }
                case 1:
                {
                    cnt++;
                    if(cnt == k)
                    {
                        return p->val;
                    }
                    stk1.pop();
                    if(p->left)
                    {
                        stk1.push(p->left);stk2.push(0);
                    }

                    break;
                }
            }
        }

        return 0;
    }
};
*/

/*
//8.10 剑指offer T45 把数组排成最小的整数
//输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

//用内置函数
bool cmp(string& x,string& y)
{
    return x + y < y + x;
}

class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        string res;
        int i = 0;

        for(i = 0;i < nums.size();i++)
        {
            strs.push_back(to_string(nums[i]));
        }

        sort(strs.begin(),strs.end(),cmp);

        for(i = 0;i < nums.size();i++)
        {
            res += strs[i];
        }

        return res;
    }
};

int main()
{
    Solution A;
    vector<int> nums = {3,30,34,5,9};
    cout << A.minNumber(nums);
}
*/

/*
//8.10 剑指offer T61 扑克牌中的顺子
// 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，
// 而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

class Solution {
public:
    bool isStraight(vector<int>& nums) {
        int cnt = 0;//记录大小王个数
        int i = 0;

        sort(nums.begin(),nums.end());

        if(nums[0] == 0)//有大小王
        {
            while(nums[i] == 0)
            {
                cnt++;
                i++;
            }
        }
        for(;i < 4;i++)
        {
            if(nums[i] == nums[i + 1]) return false;
        }

        return (nums[4] - nums[cnt]) <= 4;
    }
};

int main()
{
    Solution A;
    vector<int> nums = {0,0,1,2,5};
    cout<<A.isStraight(nums);
}
*/

/*
//8.11 剑指offer T40 最小的k个数
//输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

//快速排序
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        quickSort(arr,0,arr.size() - 1);
        vector<int> res;
        res.assign(arr.begin(),arr.begin() + k);

        return res;
    }

    void quickSort(vector<int>& arr,int l,int r)
    {
        if(l >= r) return;
        int i = l,j = r;

        //基准数arr[l]
        while(i < j)
        {
            while(i < j && arr[j] >= arr[l]) j--;
            while(i < j && arr[i] <= arr[l]) i++;
            swap(arr[i],arr[j]);
        }

        swap(arr[l],arr[i]);

        quickSort(arr,l,i - 1);
        quickSort(arr,i + 1,r);
    }
};
*/

/*
 //8.11  剑指offer T41 数据流的中位数
//  如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
//  如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
// 设计一个支持以下两种操作的数据结构：
// void addNum(int num) - 从数据流中添加一个整数到数据结构中。
// double findMedian() - 返回目前所有元素的中位数。

class MedianFinder {
public:
    priority_queue<int,vector<int>,greater<int>> A;//小顶堆，储存较大的一半
    priority_queue<int,vector<int>,less<int>> B;//大顶堆，储存较小的一半

    MedianFinder() {}

    void addNum(int num) {
        if(A.size() == B.size())
        {
            B.push(num);
            A.push(B.top());
            B.pop();
        }

        else{
            A.push(num);
            B.push(A.top());
            A.pop();
        }
    }

    double findMedian() {
        if(A.size() == B.size())
        {
            return (A.top() + B.top())/2;
        }
        else return A.top();
    }
};
*/

/*
//8.11 剑指offer T55 二叉树的深度 1
//输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;

        return max(maxDepth(root->left),maxDepth(root->right)) + 1;
    }
};
*/

/*
//8.11 剑指offer T55 平衡二叉树 2
//输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

//从上往下，时间复杂度高
class Solution1 {
public:
    int depth(TreeNode* root)
    {
        if(!root) return 0;

        return max(depth(root->left),depth(root->right)) + 1;
    }

    bool isBalanced(TreeNode* root) {
        if(!root) return true;

        return (abs(depth(root->left) - depth(root->right)) <= 1 ) && isBalanced(root->left) && isBalanced(root->right);
    }
};

//从下往上，要剪枝
class Solution2 {
public:
    bool isBalanced(TreeNode* root) {
        return recur(root) != -1;
    }

    int recur(TreeNode* root)//返回-1表示以root为根的树不是平衡二叉树，其余即返回平衡因子的大小
    {
        if(!root) return 0;

        int left = recur(root->left);
        if(left == -1) return -1;

        int right = recur(root->right);
        if(right == -1) return -1;

        return (abs(left - right) <= 1)?(max(left,right) + 1):-1;
    }
};
*/

/*
//8.12 剑指offer T64 求1+2+...+n
//求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

class Solution {
public:
    int sumNums(int n) {
        (n >= 1) && (n += sumNums(n - 1));
        return n;
    }
};

int main()
{
    Solution A;
    cout<<A.sumNums(5);
}
*/

/*
//8.12 剑指offer T68 二叉搜索树的最近公共祖先 1
//给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

//递归
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return nullptr;

        if(root->val > p->val && root->val > q->val)
        {
            return lowestCommonAncestor(root->left,p,q);
        }

        else if(root->val < p->val && root->val < q->val)
        {
            return lowestCommonAncestor(root->right,p,q);
        }

        else return root;
    }
};
*/

/*
//8.12 剑指offer T68 二叉树的最近公共祖先 2
//给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root || p == root || q == root) return root;

        TreeNode* l = lowestCommonAncestor(root->left,p,q);
        TreeNode* r = lowestCommonAncestor(root->right,p,q);

        if(l && !r) return l;
        else if(!l && r) return r;
        else if(l && r) return root;
        else return nullptr;
    }
};
*/

/*
//8.13 剑指offer T16 数值的整数次方
//实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，x^n）。不得使用库函数，同时不需要考虑大数问题。

class Solution {
public:
    double myPow(double x, long long n) {
        if(x == 0) return 0;
        if(n == 0) return 1;
        else if(n < 0) {x = 1 / x;n = -n;}

        if(n % 2 == 1)
        {

        }
        else
        {

        }
    }
};

int main()
{
    Solution A;
    cout<<A.myPow(20,-1);
}
*/

/*
//8.13 剑指offer T33 二叉搜索树后序遍历
// 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。
// 如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        if(postorder.size() == 0) return true;

        return verify(postorder,0,postorder.size() - 1);
    }

    bool verify(vector<int>& postorder,int l,int r)
    {
        if(l == r) return true;
        int i = l,j = 0;

        while(postorder[i] < postorder[r]) i++;
        j = i;
        while(postorder[j] > postorder[r]) j++;

        if(j < r) return false;

        return verify(postorder,l,j - 1) && verify(postorder,j,r);
    }
};

int main()
{
    Solution A;
    vector<int> postorder = {1,6,3,2,5};
    cout<<A.verifyPostorder(postorder);
}
*/

/*
//8.13 剑指offer T07 重建二叉树
//输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0) return nullptr;

        return recur(preorder,inorder,0,preorder.size() - 1,0,inorder.size() - 1);
    }

    TreeNode* recur(vector<int>& preorder, vector<int>& inorder,int l_in,int r_in,int l_pre,int r_pre)
    {
        if(l_in > r_in) return nullptr;

        TreeNode* head = new TreeNode(preorder[l_pre]);
        if(l_in == r_in)
        {
            return head;
        }

        int i = l_in;
        while(inorder[i] != preorder[l_pre]) i++;

        head->left = recur(preorder,inorder,l_in,i - 1,l_pre + 1,l_pre + i - l_in);
        head->right = recur(preorder,inorder,i + 1,r_in,l_pre + i + 1 - l_in,r_pre);

        return head;
    }
};

int main()
{
    Solution A;
    vector<int> preorder = {1,2};
    vector<int> inorder = {2,1};
    TreeNode *p = A.buildTree(preorder,inorder);
}
*/

/*
//8.13 剑指offer T15 二进制中1的个数
//编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量）。

class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while(n > 0)
        {
            if(n & 1) count++;
            n >>= 1;
        }

        return count;
    }
};
int main()
{
    Solution A;
    cout<<A.hammingWeight(00000000000000000000000000001011);
}
*/

/*
//8.13 剑指offer T65 不用加减乘除做加法
//写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

class Solution {
public:
    int add(int a, int b) {
        while(b != 0)
        {
            int c = (a & b) << 1;//进位
            a = a ^ b;//无进位和
            b = c;     //把进位和赋给b,下一轮加上
        }
        return a;
    }
};
*/

/*
//8.14 剑指offer T56 数组中数字出现次数 1
// 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。
// 请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int arr = nums[0],i = 0;
        int dif = 1;
        int res1 = 0,res2 = 0;

        for(i = 1;i < nums.size();i++)
        {
            arr ^= nums[i];
        }
        //找到两个数的区别
        while((arr & dif) == 0)
        {
           dif <<= 1;
        }
        //分组
        for(int num:nums)
        {
            if(num & dif) res1 ^= num;
            else res2 ^= num;
        }

        return vector<int> {res1,res2};
    }
};

int main()
{
    Solution A;
    vector<int> nums = {1,2,5,2};
    vector<int> res = A.singleNumbers(nums);
}
*/

/*
//8.14 剑指offer T56 数字中数字出现次数 2
//在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        vector<int> count(32);
        int i = 0;
        int res = 0;

        for(int num:nums)
        {
            for(i = 0;i < 32;i++)
            {
                count[i] += num & 1;
                num >>= 1;
            }
        }

        for(i = 31;i >= 0;i--)
        {
            res <<= 1;
            res |= count[i] % 3;
        }

        return res;
    }
};
*/

/*
//8.14 剑指offer T39 数组中出现次数超过一半的数字
// 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
// 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

//哈希表
class Solution1 {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int,int> map;
        int i = 0;

        for(int num:nums)
        {
            map[num]++;
        }

        for(int num:nums)
        {
            if(map[num] >= nums.size() / 2) return num;
        }

        return 0;
    }
};

//摩尔投票法
class Solution2 {
public:
    int majorityElement(vector<int>& nums) {
        int i = 1,res = nums[0],grade = 1;

        for(i = 1;i < nums.size();i++)
        {
            if(grade == 0) res = nums[i];
            if(nums[i] != res) grade--;
            else grade++;

        }
        return res;
    }
};

int main()
{
    Solution2 A;
    vector<int> nums = {2,1,2,3,2,2,4};
    cout<<A.majorityElement(nums);
}
*/

/*
//8.15 剑指offer T66 不用除法构建乘积数组
// 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积,
// 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        int i = 0,len = a.size();
        vector<int> left(len,1); //储存左乘积
        vector<int> right(len,1); //储存右乘积

        for(i = 1;i < len;i++)
        {
            left[i] = left[i - 1]*a[i - 1];
            right[len - 1 - i] = right[len - i]*a[len - i];
        }

        for(i = 0; i < len;i++)
        {
            right[i] *= left[i];
        }

        return right;
    }
};
*/

/*
//8.16 剑指offer T14 剪绳子 1
// 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
// 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

class Solution {
public:
    int cuttingRope(int n) {
        if(n == 2) return 1;
        if(n == 3) return 2;

        if(n % 3 == 1) return 4 * pow(3,(n - 4)/3);
        else if(n % 3 == 2) return 2 * pow(3,(n - 2)/3);
        else return pow(3,n/3);
    }
};

int main()
{
    Solution A;
    cout<<A.cuttingRope(23);
}
*/

/*
//8.16 剑指offer T57 和为s的连续整数数列

class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        if(target == 1 || target == 2) return vector<vector<int>> {{}};

        vector<vector<int>> res;
        target *= 2;
        int i = 0,j = 0;
        for(i = 2;i <= pow(target,0.5);i++)
        {
            if(target % i == 0 && (target / i - i) % 2 == 1)//i是target的因子且奇偶不同
            {
                vector<int> tmp;
                for(j = (target/i - i + 1)/2;j <= (target/i + i - 1)/2;j++)
                {
                    tmp.push_back(j);
                }
                res.push_back(tmp);
            }
        }
        reverse(res.begin(),res.end());

        return res;
    }
};

int main()
{
    Solution A;
    vector<vector<int>>a = A.findContinuousSequence(15);
}
*/

/*
//8.16 剑指offer T62 圆圈中剩下的数字
// 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。
// 求出这个圆圈里剩下的最后一个数字。
// 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

class Solution {
public:
    int lastRemaining(int n, int m) {
        if(n == 1) return 0;

        int res = 0;
        for(int i = 1;i < n;i++)
        {
            res = (res + m) % (i + 1);
        }

        return res;
    }
};
*/

/*
//8.16 剑指offer T31 栈的压入、弹出序列
// 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
// 例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        int len = pushed.size();
        if(len == 0) return true;

        stack<int> A;
        int i = 1,j = 0;
        A.push(pushed[0]);

        while(j < len)
        {
            if(i < len && (A.empty() || A.top() != popped[j]))
            {
                A.push(pushed[i++]);
            }
            else if(A.top() == popped[j])
            {
                A.pop();j++;
            }
            else return false;
        }
        return true;
    }
};

int main()
{
    Solution A;
    vector<int> pushed = {4,0,1,2,3},popped = {4,2,3,0,1};
    cout<<A.validateStackSequences(pushed,popped);
}
*/

/*
//8.17 剑指offer T59 滑动窗口的最大值 1
//给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(k == 0 || nums.size() == 0) return vector<int> {};
        if(k == 1) return nums;

        deque<int> deq;
        vector<int> res;int i = 0,j = 0,len = nums.size();
        deq.push_back(nums[0]);

        //初始化
        for(j = 1;j < k - 1;j++)
        {
            if(nums[j] >= deq.front()) {deq.clear();deq.push_back(nums[j]);}
            else if(nums[j] <= deq.back()) deq.push_back(nums[j]);
        }

        for(i = 0,j = k - 1;i <= len - k;i++,j++)
        {
            if(i > 0 && nums[i - 1] == deq[0]) deq.pop_front();//最大值被移开

            while (!deq.empty() && deq.back() < nums[j]) deq.pop_back();
            deq.push_back(nums[j]);

            res.push_back(deq[0]);
        }

        return res;
    }
};

int main()
{
    Solution A;
    vector<int> nums = {-7,-8,7,5,7,1,6,0};
    vector<int> res = A.maxSlidingWindow(nums,4);
    for(int r:res)
    {
        cout<<r<<' ';
    }
}
*/

/*
//8.17 剑指offer T59 队列的最大值 2
// 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
// 若队列为空，pop_front 和 max_value 需要返回 -1

class MaxQueue {
public:
    queue<int> que;
    deque<int> maxi;
    MaxQueue() {}

    int max_value() {
        if(maxi.empty()) return -1;

        return maxi.front();
    }

    void push_back(int value) {
        que.push(value);
        while(!maxi.empty() && maxi.back() < value) {maxi.pop_back();}

        maxi.push_back(value);
    }

    int pop_front() {
        if(que.empty()) return -1;

        int tmp = que.front();
        if(tmp == maxi.front()) {maxi.pop_front();}
        que.pop();

        return tmp;
    }
};
*/

/*
//8.17 剑指offer T37  序列化二叉树

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Codec {
public:
    int depth(TreeNode* root)
    {
        if(!root) return 0;
        return max(depth(root->left),depth(root->right)) + 1;
    }
    string serialize(TreeNode* root) {
        if(!root) return "[]";

        stack<TreeNode*> stk1;stack<int> stk2;
        vector<string> res(1000000000,"null");
        TreeNode* p = root;int idx = 0,num = 0;
        stk1.push(root);stk2.push(0);

        while(!stk1.empty())
        {
            p = stk1.top();idx = stk2.top();
            res[idx] = p->val + '0';
            stk1.pop();stk2.pop();

            if(p->right) {stk1.push(p->right);stk2.push(2 * idx + 2);}
            if(p->left) {stk1.push(p->left);stk2.push(2 * idx + 1);}
        }

        for(num = pow(2,depth(root)) - 1;num >= 0;num--)
        {
            if(res[num] != "null") break;
        }
        res.erase(res.begin() + num + 1,res.end());

        string ans;
        ans +="[";
        for(int i = 0;i <= num;i++)
        {
            ans+=res[i];
            if(i != num) ans+=",";
        }
        ans += "]";

        return ans;
    }


    TreeNode* deserialize(string data) {
        if(data == "[]") return nullptr;

        vector<string> node;//用来储存分开的节点
        int i = 1,j = 1;

        while(j < data.size())
        {
            if(data[j + 1] == ',' || data[j + 1] == ']')
            {
                node.push_back(data.substr(i,j - i + 1));
                j += 2;
                i = j;
            }
            else j++;
        }
        if(node[0] == "null") return nullptr;

        vector<TreeNode*> arr(node.size());
        for(i = 0;i < node.size();i++)
        {
            if(node[i] != "null") arr[i] = new TreeNode(0);
            else arr[i] = nullptr;
        }

        for(int i = node.size() - 1;i >= 1;i--)
        {
            if(arr[i]) arr[i]->val = node[i][0] - '0';

            if(arr[(i - 1)/2])
            {
                if(i % 2 == 1) {arr[(i - 1)/2]->left = arr[i];}
                else {arr[(i - 1)/2]->right = arr[i];}
            }
        }
        arr[0]->val = node[0][0] - '0';

        return arr[0];
    }
};

int main()
{
    Codec A;
    TreeNode* root = new TreeNode(1);TreeNode* l = new TreeNode(2);TreeNode* r = new TreeNode(3);TreeNode* rl = new TreeNode(4);TreeNode* rr = new TreeNode(5);
    r->left = rl;r->right = rr;root->left = l;root->right = r;

    TreeNode* tmp  = A.deserialize(A.serialize(root));
}
*/

/*
//8.18 剑指offer T49 丑数
//我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

class Solution {
public:
    int nthUglyNumber(int n) {
        if(n <= 6) return n;

        vector<int> ugly(n + 1);
        for(int i = 1;i < 7;i++)
        {
            ugly[i] = i;
        }

        int a = 4;//*2大于6的第一个数的索引
        int b = 3;//*3大于6的第一个数的索引
        int c = 2;//*5大于6的第一个数的索引
        int i = 7;
        while(i <= n)
        {
            ugly[i] = min(ugly[a]*2,min(ugly[b]*3,ugly[c]*5));
            while(ugly[a]*2 <= ugly[i]) a++;
            while(ugly[b]*3 <= ugly[i]) b++;
            while(ugly[c]*5 <= ugly[i]) c++;
            i++;
        }

        return ugly[n];
    }
};

int main()
{
    Solution A;
    cout<<A.nthUglyNumber(12);
}
*/

/*
//8.19 剑指offer T60 n个骰子的点数
// 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
// 你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> res(6,1);
        int i = 0;

        for(i = 2;i <= n;i++)
        {
            vector<double> tmp(5*i + 1);
            int j = 0;tmp[0] = 1;

            for(j = 1;j <= 5;j++)
            {
                tmp[j] = tmp[j - 1] + res[j];
            }

            for(j = 6;j <= 5*i - 5;j++)
            {
                tmp[j] = tmp[j - 1] + res[j] - res[j - 6];
            }

            for(j = 5*i - 4;j <= 5*i;j++)
            {
                tmp[j] = tmp[j - 1] - res[j - 6];
            }

            res = tmp;
        }

        for(i = 0;i <= 5*n;i++)
        {
            res[i] /= pow(6,n);
        }
        return res;
    }
};

int main()
{
    Solution A;
    vector<double> res = A.dicesProbability(3);
}
*/

/*
//8.19 剑指offer T50 第一个只出现一次的字符
// 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
// 输入：s = "abaccdeff"
// 输出：'b'

class Solution {
public:
    char firstUniqChar(string s) {
        int i = 0,len = s.size();

        unordered_map<char,bool> map;
        for(i = 0;i < len;i++)
        {
            map[s[i]] = (map.find(s[i]) == map.end());
        }

        for(i = 0;i < len;i++)
        {
            if(map[s[i]]) return s[i];
        }

        return ' ';
    }
};
*/

/*
//8.19 剑指offer T14 剪绳子 2
// 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1]。
// 请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
// 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

const int MAX = 1e9+7;

class Solution {
public:
    //快速幂 + 取模
    long long power(long long a,long long b)
    {
        if(b == 0) return 1;
        if(b == 1) return a % MAX;

        if(b % 2 == 0) return power((a*a) % MAX,b/2);
        else return (a * power((a*a) % MAX,b/2) % MAX);
    }

    int cuttingRope(int n) {
        if(n == 2) return 1;
        if(n == 3) return 2;

        if(n % 3 == 1) return (4 * power(3,(n - 4)/3)) % MAX;
        else if(n % 3 == 2) return (2 * power(3,(n - 2)/3)) % MAX;
        else return power(3,n/3);
    }
};
*/

/*
//8.20 剑指offer T38 字符串的排列
// 输入一个字符串，打印出该字符串中字符的所有排列。你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

class Solution1 {
public:
    vector<string> permutation(string s) {
        int len = s.size();
        vector<string> res;

        if(len == 1)
        {
            res.push_back(s);
            return res;
        }
        else
        {
            set<int> st;//用于剪枝
            for(int i = 0;i < len;i++)
            {
                if(st.find(s[i]) != st.end()) continue;

                string s1 = s;
                string tmp = s.erase(i,1);
                s = s1;
                st.insert(s[i]);

                vector<string> dps = permutation(tmp);

                for(string str:dps)
                {
                    string tmp2;
                    tmp2.push_back(s[i]);

                    res.push_back(tmp2 + str);
                }
            }

            return res;
        }
    }
};

class Solution2 {
public:
    vector<string> res;
    vector<string> permutation(string s) {
        dfs(s,0);
        return res;
    }

    void dfs(string s,int x)
    {
        if(x == s.size() - 1)
        {
            res.push_back(s);
            return;
        }

        set<int> st;
        for(int i = x;i < s.size();i++)
        {
            if(st.find(s[i]) != st.end()) continue;
            st.insert(s[i]);

            swap(s[x],s[i]);//固定第x位为s[i]
            dfs(s,x + 1);
            swap(s[x],s[i]);//回溯算法
        }
    }
};

int main()
{
    Solution2 A;
    string s = "aab";
    vector<string> res = A.permutation(s);

    for(string s:res)
    {
        cout<<s<<' ';
    }
}
*/

/*
//8.21 剑指offer T51 数组中的逆序对
//在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

class Solution {
public:
    int reversePairs(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        return recur(nums, 0, nums.size() - 1);
    }

    int recur(vector<int>& nums, int left, int right)
    {
        if (left == right) return 0;

        int mid = (left + right) / 2;

        int a = recur(nums, left, mid);
        int b = recur(nums, mid + 1, right);

        int c = Merge(nums, left, mid, right);

        return a + b + c;
    }

    int Merge(vector<int>& nums, int left, int mid, int right)
    {
        if (left == right) return 0;
        if (right - left == 1)
        {
            if (nums[left] > nums[right]) { swap(nums[left], nums[right]); return 1; }
            else return 0;
        }

        int cnt = 0, i = left, j = mid + 1;
        int cnt1 = 0;//阶段性计数

        vector<int> tmp;

        while (i <= mid && j <= right)
        {
            if (nums[i] > nums[j])
            {
                cnt1++;         //阶段性逆序对个数
                tmp.push_back(nums[j]);
                j++;
            }
            else
            {
                cnt += cnt1 * (mid - i + 1);
                cnt1 = 0;
                tmp.push_back(nums[i]);
                i++;
            }
        }

        if (i <= mid)
        {
            cnt += cnt1 * (mid - i + 1);
            while (i <= mid) { tmp.push_back(nums[i]); i++; }
        }
        if (j <= right)
        {
            while (j <= right) { tmp.push_back(nums[j]); j++; }
        }

        for (i = left; i <= right; i++)
        {
            nums[i] = tmp[i - left];
        }

        return cnt;
    }
};

int main()
{
    Solution A;
    vector<int> nums = { 1,3,2,5,6,8 };
    cout << A.reversePairs(nums);
}
*/

/*
//8.22 剑指offer T41 数字序列的某一位数字
// 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
// 请写一个函数，求任意第n位对应的数字。

class Solution {
public:
    int findNthDigit(int n) {
        int i = 0;

        while(fun(i) <= n) i++;
        i--;
        int real = pow(10,i);

        int a = n - fun(i) + 1;
        int b = a / (i + 1); //第几个（i+1）位数
        int c = a - b*(i + 1); //的第几位
        if(a % (i + 1) != 0) b++;

        real += (b - 1);

        return int(real/(pow(10,c))) % 10;
    }

    int fun(int n)
    {
        return (n - 1/9.0)*pow(10,n) + 10/9.0;
    }
};

int main()
{
    Solution A;
    cout<<A.findNthDigit(15);
}
*/

/*
//9.1 T337 打家劫舍 3
// 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。
// 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。
// 一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
// 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。


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
    int rob(TreeNode* root) {
        recur(root);
        return max(map1[root],map2[root]);
    }

    //记忆化递推：增加一个哈希表，来储存已经得出的结果
    unordered_map<TreeNode*,int> map1,map2; //后序遍历，从下往上计算，map1存储选的，map2存储不选的

    void recur(TreeNode* root)
    {
        if(!root) return;

        recur(root->left);recur(root->right);//后序遍历

        map1[root] = root->val + map2[root->left] + map2[root->right];//向下递推
        map2[root] = max(map1[root->left],map2[root->left]) + max(map1[root->right],map2[root->right]);
    }
};
*/

/*
int BFS(vector<vector<int>>& map,vector<int>& start,vector<int>& end,int M,int N,int M1,int M2)
{
    int i,j;
    vector<int> dir1 = {M1,-M1,M1,-M1,M2,-M2,-M2,M2},dir2 = {M2,-M2,-M2,M2,M1,-M1,M1,-M1};
    vector<vector<bool>> unset(M,vector<bool>(N,false));
    queue<vector<int>> que;//用坐标来储存节点
    que.push(start);//初始状态入队
    unset[start[0]][start[1]] = true;//已经访问过
    int res = 0;

    while(!que.empty())
    {
        res++;//相当于深度+1
        for(i = 0;i < que.size();i++)
        {
            vector<int> top = que.front();que.pop();
            if(top == end) return res;

            for(j = 0;j < 8;j++)
            {
                int a = top[0] + dir1[j],b = top[1] + dir2[j];
                if(a >= 0 && a < M && b >= 0 && b < N && (map[a][b] == 1 || map[a][b] == 4) && !unset[a][b])//八个方向的遍历
                {
                    que.push({a,b});
                    unset[a][b] = true;
                }
            }
        }
    }
    return res;
}

int main()
{
    int M,N,M1,M2;
    int i = 0,j = 0;
    cin >> M >> N >> M1 >> M2;

    vector<int> start,end;
    vector<vector<int>> map(M,vector<int>(N));
    for(i = 0;i < M;i++)
    {
        for(j = 0;j < N;j++)
        {
            cin>>map[i][j];
            if(map[i][j] == 3) start = {i,j};
            else if(map[i][j] == 4) end = {i,j};
        }
    }

    cout << BFS(map,start,end,M,N,M1,M2);
}
*/

/*
//7.3 T239 望远镜中最高的海拔
//科技馆内有一台虚拟观景望远镜，它可以用来观测特定纬度地区的地形情况。该纬度的海拔数据记于数组 heights ，
//其中 heights[i] 表示对应位置的海拔高度。请找出并返回望远镜视野范围 limit 内，可以观测到的最高海拔值。

class Solution {
public:
    vector<int> maxAltitude(vector<int>& heights, int limit) {
        vector<int> res;
        deque<int> q;

        if(heights.empty()) return res; //空队列情况

        for(int i = 0; i <= heights.size() - 1; i++)
        {
            if(i - limit >= 0 && q[0] == heights[i - limit]) //要被剔除
            {
                q.pop_front();
            }
            while(!q.empty() && q.back() < heights[i]) //将列表中所有小于加入元素的数删掉，以保证队列的递减性
            {
                q.pop_back();
            }
            q.push_back(heights[i]); //加入新元素

            if(i >= limit - 1) res.push_back(q[0]);
        }

        return res;
    }
};
*/

/*
//7.8 T10 模糊搜索验证
// 请设计一个程序来支持用户在文本编辑器中的模糊搜索功能。用户输入内容中可能使用到如下两种通配符：
// '.' 匹配任意单个字符。
// '*' 匹配零个或多个前面的那一个元素。
// 请返回用户输入内容 input 所有字符是否可以匹配原文字符串 article。

class Solution {
public:
    bool articleMatch(string s, string p) {
        int n = s.size(); int m = p.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false)); // 初始化dp全都为false
        dp[0][0] = true; //空序列对空序列，可以匹配

        for(int j = 1; j <= m; j++)  // 计算dp[i][j]的值
        {
            for(int i = 0;i <= n; i++)
            {
                if(p[j - 1] == '*')
                {
                    bool a = (j >= 2) && dp[i][j - 2]; // 0个*前的字母是否使其成立
                    bool b = (i >= 1) && (j >= 2) && dp[i - 1][j] && ((p[j - 2] == s[i - 1]) || p[j - 2] == '.');
                    //  >=1个*前的字母是否使其成立，或者是特殊情况p[j - 2] = '.'
                    dp[i][j] = a || b;
                }
                else if(p[j - 1] == '.') dp[i][j] = (i >= 1) && dp[i - 1][j - 1]; // 是'.'
                else dp[i][j] = (i >= 1) && dp[i - 1][j - 1] && (s[i - 1] == p[j - 1]); // 是字母
            }
        }

        return dp[n][m];
    }
};
*/

/*
// 7.9 T12 合并有序链表
//  给定两个以 有序链表 形式记录的训练计划 l1、l2，分别记录了两套核心肌群训练项目编号，请合并这两个训练计划，
//  按训练项目编号 升序 记录于链表并返回。
//  注意：新链表是通过拼接给定的两个链表的所有节点组成的。

struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution
{
public:
    ListNode *trainningPlan(ListNode *l1, ListNode *l2)
    {
        ListNode *p1 = l1, *p2 = l2;        // 两个指针，用于遍历
        ListNode *combine = new ListNode(); // 合并后的链表
        ListNode *l = combine;              // 遍历合并后的链表
        l = combine;
        while (!p1 && !p2) // 当两个指针都没到末尾时
        {
            if (p1->val <= p2->val)
            {
                l->val = p1->val;
                p1 = p1->next;
            }
            else
            {
                l->val = p2->val;
                p2 = p2->next;
            }
            l->next = new ListNode();
            l = l->next;
        }
        // 以下两个while只会被执行一遍。
        while (p1)
        {
            l->next = p1;
        }
        while (p2)
        {
            l->next = p2;
        }
        return combine;
    }
};

int main()
{
}
*/


/*
// T1229 
// 给定两个人的空闲时间表：slots1 和 slots2，以及会议的预计持续时间 duration，请你为他们安排 时间段最早 且合适的会议时间。
// 如果没有满足要求的会议时间，就请返回一个 空数组。
// 「空闲时间」的格式是 [start, end]，由开始时间 start 和结束时间 end 组成，表示从 start 开始，到 end 结束。 
// 题目保证数据有效：同一个人的空闲时间不会出现交叠的情况，也就是说，对于同一个人的两个空闲时间 [start1, end1] 
// 和 [start2, end2]，要么 start1 > end2，要么 start2 > end1。
bool cmp1(vector<int> a, vector<int> b)
{
    return a[0] <= b[0];
}
class Solution {
public:
    vector<int> minAvailableDuration(vector<vector<int>>& slots1, vector<vector<int>>& slots2, int duration) {
        int num1 = slots1.size(), num2 = slots2.size();

        sort(slots1.begin(), slots1.end(), cmp1); //升序排列，当然这里不用cmp也行，而且会快很多
        sort(slots2.begin(), slots2.end(), cmp1);
        int i = 0, j = 0; //用双指针
        while((i < num1) && (j < num2))
        {
            if(min(slots1[i][1], slots2[j][1]) - max(slots1[i][0], slots2[j][0]) >= duration)
            {
                return {max(slots1[i][0], slots2[j][0]), max(slots1[i][0], slots2[j][0]) + duration};
            }
            if(slots1[i][1] >= slots2[j][1]) j++;
            else i++;
        }
        return {};
    }
};
*/