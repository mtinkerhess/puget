# Puget

I really like writing Python, but it has its quirks. One day I got to thinking, what if Python used curly braces instead of indentation? And used semicolons instead of newlines? Sometimes it's good to be explicit about stuff like that. Then I thought, what if you didn't have to manually include "self" as a member on every member function? What if there was only one type of class inheritance?

I've had fun programming in Lisp too, but it also has its quirks. I really like being able to set a breakpoint and look at the state of a program's execution, which can be tricky in Lisp. The parentheses can get a little unwieldy, and there are too many macros and weird special forms that seem to be there just to make every edge case as elegant as possible. So I thought, well, if the functions are all in prefix notation anyway, and you're pretty much always using a fixed number of arguments for a function, then you don't really need any parentheses, do you?

Then I read about GolfScript and thought, shoot, how hard could it be to implement a stack-oriented prefix-notation language?

So Puget is my experiment to see how well this idea turns out. I don't claim to be a programming language expert, but I hope by working on this project I'll learn something about implementing languages. Right now I'm implementing it in Python, and once I'm more or less happy with how that works, if I'm still feeling motivated, I might implement it in C++.

## Getting Started

Using Python 3.3, just run python/Puget/Puget.py and you should get a REPL. Use the Visual Studio project with Python Tools For Visual Studio if you want.

## Code

Most of this code works! Some of it doesn't yet (member functions on lists), and some of it will look better in the future (syntactic sugar for declaring classes and functions, removing superfluous semicolons).

    + 2 * 3 4;
    # Evaluates to 14
    
    + * 2 3 4;
    # Evaluates to 10
    
    let x 10;
    let y 3;
    % x y;
    # 1
    
    let foo fun { print 'hello!' };
    foo;
    # Prints 'hello!'
    
    let factorial fun x
    {
        if = x 1
        {
            1
        }
        else
        {
            * x factorial - x 1
        }
    };
    
    factorial 4;
    # 24
    
    # Base class
    let Foo class
    {
        # Constructor
        let init fun
        {
            let this.val 23
        };
    
        let inc fun
        {
            let this.val + this.val 1
        }
    
        # Member function with argument
        let inc_by fun x
        {
            let this.val + this.val x
        }
    };
    
    # Derived class, inherits Foo
    let Bar class Foo
    {
        # Overridden constructor and inc function
        let init fun
        {
            let this.val 34
        };
    
        let inc fun
        {
            let this.val + this.val 2
        }
    };
    
    let f new Foo;
    f.inc;
    f.val;
    # 24
    
    let b new Bar;
    b.inc;
    b.inc_by 5;
    b.val;
    # 41
    
    # Solutions to Project Euler problems
    # Based on solutions from https://wiki.python.org/moin/ProblemSets/Project%20Euler%20Solutions
    
    let filter fun list predicate
    {
        # Initialize ret to empty list
    	let ret [];
    	for list fun item
    	{
    		if predicate item
    		{
    			ret.append item
    		}
    	};
    	ret
    };
    
    let sum fun list
    {
    	let ret 0;
    	for list fun item
    	{
    		let ret + ret item
    	};
    	ret
    };
    
    let range fun length
    {
    	let ret [];
    	while < ret.size length
    	{
    		ret.append ret.size
    	};
    	ret
    };
    
    let euler_1_helper fun max
    {
    	sum filter range max fun n { or = 0 % n 3 = 0 % n 5 }
    };
    
    let euler_1 fun
    {
    	euler_1_helper 1000
    };
    
    let fibs_under fun max
    {
    	let ret [1, 1];
    	while < (ret.get_at - ret.size 1) max
    	{
    		ret.append + (ret.get_at - ret.size 1) (ret.get_at - ret.size 2);
    	};
    	ret.pop;
    	ret
    };
    
    let filter_even fun seq
    {
    	filter seq fun item { = 0 % item 2 }
    };
    
    let euler_2_helper fun max
    {
    	sum filter_even fibs_under max
    };
    
    let euler_2 fun
    {
    	euler_2_helper 4000000
    };
    
    let max fun items
    {
    	let ret null;
    	for items fun item
    	{
    		if > item ret
    		{
    			let ret item
    		}
    	};
    	ret
    };
    
    let map fun items mapping
    {
    	let ret [];
    	for items fun item
    	{
    		ret.append mapping item
    	};
    	ret
    };
    
    let any fun items predicate
    {
    	let ret false;
    	for items fun item
    	{
    		if predicate item
    		{
    			let ret true;
    			break
    		}
    	};
    	ret
    };
    
    # based on http://stackoverflow.com/questions/14138053/project-euler-3-with-python-most-efficient-method
    let euler_3_helper fun n
    {
    	let i 2;
    	while < * i i n
    	{
    		while = 0 % n i
    		{
    			let n / n i;
    		};
    		let i + i 1
    	};
    	n
    };
    
    let euler_3 fun
    {
    	euler_3_helper 600851475143
    };
